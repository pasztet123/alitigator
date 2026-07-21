"""Model-first legal query planning with an explicitly bounded legacy fallback."""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from typing import Iterable, Optional

from pydantic import BaseModel, Field, ValidationError

from app.model_gateway import (
    ModelGateway,
    ModelGatewayError,
    ModelSchemaError,
    ModelTechnicalError,
    ModelTransportError,
    ModelUnavailableError,
)

from .schemas import (
    Clarification,
    Fact,
    FallbackCandidate,
    FallbackReason,
    FallbackTrace,
    LegalIssue,
    LegalResearchPlan,
    MissingFact,
    QueryFamily,
    ResearchIntent,
    SourceSpan,
)


PLANNER_PROMPT_VERSION = "legal_query_planner_v2_3"
DEFAULT_PLANNER_MODEL = "gpt-5.6-terra"


PLANNER_SYSTEM_PROMPT = """\
You are a planning component for research in Polish tax law. Return only the
LegalResearchPlan required by the supplied schema. Do not answer the legal
question, predict its outcome, or state a final legal conclusion.

Extract only facts grounded in the user's exact question. Every explicit or
language-inferred fact must point to a half-open character source_span in that
question; preserve the user's literal wording in `value` for explicit facts.
Do not assume a missing fact. Put it in missing_facts instead. A provision
concept suggested by you is only a retrieval hint, never evidence.

Separate independent legal issues and distinguish taxpayer/entity roles,
transactions, payments, dates and jurisdictions. Include positive and negative
fact constraints that help distinguish legally adjacent cases. For each issue,
name the concrete legal mechanism and create useful query families, without
inventing document IDs or judgment signatures. When you can identify a
specific statutory candidate, include a separate primary-law query family
whose query contains the act/domain and exact article reference. Do not use a
general cost rule as a substitute for a special adjustment, exclusion,
exemption, timing, payment-channel or procedural mechanism described by the
facts. Unless the user's request clearly narrows the task, request primary law,
tax interpretations and judgments.

Ask at most three clarification questions. Ask only when an absent fact can
change the legal result, materially change retrieval, or distinguish two close
legal mechanisms. Otherwise continue with the fact marked missing. Confidence
measures confidence in this research plan, not confidence in a legal answer.

The input may contain deterministic named-institution locks. They come from a
versioned dictionary before this model call. Treat them as mandatory retrieval
constraints: do not replace them with a broad mechanism or remove them. Keep
any independent model hypothesis in `model_inferred_institutions`; a lock is a
retrieval signal, never a legal conclusion.
"""

ISSUE_LOCATOR_SYSTEM_PROMPT = """\
You repair the issue-identification stage of a Polish tax-law research plan.
Return only the requested structured issue locator output. Do not answer the
legal question and do not state a legal conclusion. Split independent
transactions and legal mechanisms. For every issue provide a specific label,
tax domain, mechanism, useful primary-law provision candidates and query
families for primary law, interpretations and judgments. Provision candidates
are retrieval hints only and are not evidence. Never return labels such as
"general tax issue" when the question describes a concrete mechanism.
"""


class IssueLocatorOutput(BaseModel):
    issues: list[LegalIssue] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


class PlannerValidationError(ValueError):
    """The plan is structurally valid JSON but not grounded in the question."""


@dataclass(frozen=True)
class PlannerOutcome:
    plan: LegalResearchPlan
    fallback_trace: FallbackTrace
    planner_model: str
    primary_succeeded: bool


def _slug(value: str, *, fallback: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return (normalized[:80] or fallback).strip("_")


def _dedupe(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = " ".join(str(value).split())
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


def _literal_terms(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[0-9A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]+", value.lower())
        if len(token) >= 3
    }


def validate_plan_grounding(plan: LegalResearchPlan, question: str) -> LegalResearchPlan:
    """Bind a structured plan to the exact input and reject unsupported facts.

    This validation is deliberately deterministic. It does not try to decide
    whether a legal issue is correct; it only enforces provenance invariants.
    """

    if not question or not question.strip():
        raise PlannerValidationError("question cannot be empty")

    for fact in plan.facts:
        if fact.status == "missing":
            continue
        span = fact.source_span
        if span is None:  # guarded by the schema, retained for defensive clarity
            raise PlannerValidationError(f"fact {fact.fact_id!r} has no source span")
        if span.source_id != "user_question":
            raise PlannerValidationError(
                f"fact {fact.fact_id!r} references a source other than the user question"
            )
        if span.start < 0 or span.end > len(question) or span.end <= span.start:
            raise PlannerValidationError(f"fact {fact.fact_id!r} has an invalid source span")
        source_text = question[span.start : span.end]
        if not source_text.strip():
            raise PlannerValidationError(f"fact {fact.fact_id!r} points to empty text")
        if span.quote is not None and span.quote != source_text:
            raise PlannerValidationError(f"fact {fact.fact_id!r} has a mismatched source quote")
        # Explicit values must retain at least one material token from their
        # cited wording. Canonical/inferred labels use the separate inferred
        # status and are still anchored by the span.
        if fact.status == "explicit":
            value_terms = _literal_terms(fact.value)
            source_terms = _literal_terms(source_text)
            if value_terms and source_terms and not value_terms.intersection(source_terms):
                raise PlannerValidationError(
                    f"explicit fact {fact.fact_id!r} is not supported by its source span"
                )

    # Keep the public Model→RAG→Model field names and the first v2 field names
    # losslessly synchronized while the additive migration is in progress.
    payload = plan.model_dump(mode="python")
    for issue in payload.get("issues", []):
        issue["possible_provision_concepts"] = _dedupe([
            *issue.get("possible_provision_concepts", []),
            *issue.get("possible_legal_concepts", []),
            *issue.get("possible_provision_hints", []),
        ])
        issue["positive_fact_constraints"] = _dedupe([
            *issue.get("positive_fact_constraints", []),
            *issue.get("positive_constraints", []),
        ])
        issue["negative_fact_constraints"] = _dedupe([
            *issue.get("negative_fact_constraints", []),
            *issue.get("negative_constraints", []),
        ])
    public_questions = payload.get("clarification_questions", [])[:3]
    if public_questions:
        payload["clarification"] = {"should_ask": True, "questions": public_questions}
        payload["should_ask_clarification"] = True
    elif payload.get("clarification", {}).get("questions"):
        payload["clarification_questions"] = payload["clarification"]["questions"][:3]
        payload["should_ask_clarification"] = bool(payload["clarification_questions"])

    # The input is authoritative. A provider cannot rewrite the question.
    validated = LegalResearchPlan.model_validate({**payload, "user_query": question})
    return _materialize_provision_query_families(validated)


_PLANNER_PROVISION_RE = re.compile(
    r"\bart\.\s*\d+[a-z]*"
    r"(?:\s*(?:ust\.\s*\d+[a-z]*|§\s*\d+[a-z]*))?"
    r"(?:\s*pkt\s*\d+[a-z]*)?"
    r"(?:\s*lit\.\s*[a-z])?",
    re.IGNORECASE,
)
_PLANNER_DOMAIN_RE = re.compile(
    r"\b(CIT|PIT|VAT|UFR|PCC|SD|ORDYNACJA|OP|AKCYZA|EXCISE|PP)\b",
    re.IGNORECASE,
)


def _materialize_provision_query_families(
    plan: LegalResearchPlan,
) -> LegalResearchPlan:
    """Turn model provision hypotheses into executable primary-law queries.

    This does not introduce legal knowledge. It only makes already declared
    model hints operational, preventing retrieval from falling back to the
    entire user question when the model supplied an article but omitted the
    corresponding query-family object.
    """

    changed = False
    issues: list[LegalIssue] = []
    for issue in plan.issues:
        queries = list(issue.query_families)
        existing = {
            " ".join(item.query.casefold().split())
            for item in queries
            if item.lane in {"primary_law", "both"}
        }
        candidates = _dedupe(
            [
                *issue.possible_provision_concepts,
                *issue.possible_legal_concepts,
                *issue.possible_provision_hints,
            ]
        )
        domains = _dedupe([value.upper() for value in issue.tax_domains])
        for candidate in candidates:
            reference = _PLANNER_PROVISION_RE.search(candidate)
            if reference is None:
                continue
            domain_match = _PLANNER_DOMAIN_RE.search(candidate)
            domain = domain_match.group(1).upper() if domain_match else ""
            if not domain and len(domains) == 1:
                domain = domains[0]
            if not domain:
                continue
            query = f"{domain} {' '.join(reference.group(0).split())}"
            normalized = " ".join(query.casefold().split())
            if normalized in existing:
                continue
            queries.append(
                QueryFamily(
                    family="explicit_provision_reference",
                    query=query,
                    lane="primary_law",
                    origin="model",
                )
            )
            existing.add(normalized)
            changed = True
        issues.append(issue.model_copy(update={"query_families": queries}))
    return plan.model_copy(update={"issues": issues}) if changed else plan


class LegacyFallbackPlanner:
    """Adapter around the existing rule-based planner.

    It is intentionally unavailable as an implicit primary planner. Callers
    must provide an allowed fallback reason, and every rule/hint is returned in
    :class:`FallbackTrace`. It has no field or API capable of emitting a final
    legal conclusion.
    """

    allowed_reasons: frozenset[str] = frozenset(
        {
            "planner_timeout",
            "provider_unavailable",
            "invalid_schema",
            "low_confidence",
            "insufficient_recall",
            "forced",
        }
    )

    def plan(
        self,
        question: str,
        *,
        reason: FallbackReason,
        target_date: Optional[str] = None,
        base_plan: Optional[LegalResearchPlan] = None,
        primary_error: Optional[str] = None,
    ) -> tuple[LegalResearchPlan, FallbackTrace]:
        if reason not in self.allowed_reasons:
            raise ValueError(f"unsupported fallback reason: {reason}")
        if not question or not question.strip():
            raise ValueError("question cannot be empty")

        # These imports are lazy so the model-first path neither evaluates nor
        # depends on legacy query rules during normal operation.
        from app.hybrid_authority_rag import (
            _heuristic_clarifier_questions,
            build_fact_graph,
            build_issue_graph,
            classify_legal_research_intent,
        )

        legacy_intent = classify_legal_research_intent(question)
        legacy_facts = build_fact_graph(question)
        legacy_issues = build_issue_graph(question, legacy_intent, legacy_facts)
        legacy_questions = _heuristic_clarifier_questions(question, legacy_intent)

        rules = [
            "legacy.classify_legal_research_intent",
            "legacy.build_fact_graph",
            "legacy.build_issue_graph",
        ]
        if legacy_questions:
            rules.append("legacy.heuristic_clarifier_questions")

        # V2Schema trims string fields. Keep a full-question provenance span
        # inside the non-whitespace bounds so its stored quote remains exactly
        # equal to ``question[start:end]`` during deterministic validation.
        full_span_start = len(question) - len(question.lstrip())
        full_span_end = len(question.rstrip())
        full_span = SourceSpan(
            start=full_span_start,
            end=full_span_end,
            quote=question[full_span_start:full_span_end],
            source_id="user_question",
        )
        inferred: list[Fact] = []
        fact_values = (
            ("role", legacy_facts.roles),
            ("transaction", legacy_facts.transactions),
            ("payment", legacy_facts.payments),
            ("jurisdiction", legacy_facts.jurisdictions),
            ("relationship", legacy_facts.relationships),
            ("date", legacy_facts.dates),
        )
        used_fact_ids: set[str] = set()
        for category, values in fact_values:
            for index, value in enumerate(values, start=1):
                candidate_id = f"fallback_{category}_{_slug(value, fallback=str(index))}"
                while candidate_id in used_fact_ids:
                    candidate_id = f"{candidate_id}_{index}"
                used_fact_ids.add(candidate_id)
                inferred.append(
                    Fact(
                        fact_id=candidate_id,
                        subject="case",
                        role=value if category == "role" else "case",
                        predicate=category,
                        value=value,
                        status="inferred_from_language",
                        source_span=full_span,
                    )
                )

        missing_facts = [
            MissingFact(
                fact_id=f"fallback_missing_{_slug(item.id, fallback=str(index))}",
                question=item.question,
                materiality="retrieval_relevant",
            )
            for index, item in enumerate(legacy_questions[:3], start=1)
        ]

        fallback_candidates: list[FallbackCandidate] = []
        converted_issues: list[LegalIssue] = []
        for index, issue in enumerate(legacy_issues, start=1):
            issue_id = issue.issue_id or f"fallback_issue_{index}"
            concepts: list[str] = []
            query_families = [
                QueryFamily(
                    family="natural_language",
                    query=issue.query or question,
                    lane="both",
                    origin="fallback",
                )
            ]
            fallback_candidates.append(
                FallbackCandidate(
                    candidate_type="query_hint",
                    value=issue.query or question,
                    issue_id=issue_id,
                )
            )
            if issue.mechanism:
                concepts.append(issue.mechanism)
                query_families.append(
                    QueryFamily(
                        family="legal_concept",
                        query=issue.mechanism.replace("_", " "),
                        lane="both",
                        origin="fallback",
                    )
                )
            for tax, provision in issue.preferred_targets:
                hint = " ".join(item for item in (tax, provision) if item).strip()
                if not hint:
                    continue
                concepts.append(hint)
                query_families.append(
                    QueryFamily(
                        family="explicit_provision_reference",
                        query=hint,
                        lane="primary_law",
                        origin="fallback",
                    )
                )
                fallback_candidates.append(
                    FallbackCandidate(
                        candidate_type="provision_hint",
                        value=hint,
                        issue_id=issue_id,
                    )
                )

            converted_issues.append(
                LegalIssue(
                    issue_id=issue_id,
                    label=issue.label or "General tax issue",
                    tax_domains=[issue.tax] if issue.tax else [],
                    legal_mechanism=issue.mechanism or "general_tax_analysis",
                    taxpayer_roles=list(legacy_facts.roles),
                    transactions=list(legacy_facts.transactions),
                    payments=list(legacy_facts.payments),
                    jurisdictions=list(legacy_facts.jurisdictions),
                    relevant_dates=list(legacy_facts.dates),
                    possible_provision_concepts=_dedupe(concepts),
                    positive_fact_constraints=list(legacy_facts.known_facts),
                    negative_fact_constraints=[issue.contrast] if issue.contrast else [],
                    requested_source_types=["statute", "interpretation", "judgment"],
                    query_families=query_families,
                    priority=issue.priority if issue.priority in {"high", "medium", "low"} else "medium",
                )
            )

        fallback_plan = LegalResearchPlan(
            user_query=question,
            intent=ResearchIntent(
                mode=legacy_intent.answer_mode,
                needs_normative_answer=legacy_intent.needs_normative_answer,
                needs_interpretations=True,
                needs_case_law=True,
                needs_conflict_analysis=legacy_intent.needs_conflict_analysis,
                needs_calculations=legacy_intent.needs_calculations,
            ),
            target_date=target_date,
            facts=inferred,
            missing_facts=missing_facts,
            issues=converted_issues,
            clarification=Clarification(
                should_ask=bool(missing_facts),
                questions=[item.question for item in missing_facts],
            ),
            # This is confidence in the heuristic plan, deliberately below a
            # normal model-first threshold and never a legal confidence score.
            confidence=0.35,
        )
        plan = self._merge(base_plan, fallback_plan) if base_plan is not None else fallback_plan
        plan = validate_plan_grounding(plan, question)
        return plan, FallbackTrace(
            fallback_used=True,
            fallback_reason=reason,
            fallback_rules=rules,
            fallback_candidates_added=fallback_candidates,
            primary_planner_error=primary_error,
        )

    @staticmethod
    def _merge(primary: LegalResearchPlan, fallback: LegalResearchPlan) -> LegalResearchPlan:
        """Add only missing research hints; never overwrite model conclusions.

        Neither input can contain a conclusion by schema. The primary plan's
        confidence and fact interpretation remain authoritative.
        """

        facts = list(primary.facts)
        known_fact_ids = {item.fact_id for item in facts}
        facts.extend(item for item in fallback.facts if item.fact_id not in known_fact_ids)

        missing = list(primary.missing_facts)
        missing_ids = {item.fact_id for item in missing}
        missing.extend(item for item in fallback.missing_facts if item.fact_id not in missing_ids)

        issues = list(primary.issues)
        issue_ids = {item.issue_id for item in issues}
        issues.extend(item for item in fallback.issues if item.issue_id not in issue_ids)

        questions = _dedupe(
            [*primary.clarification.questions, *fallback.clarification.questions]
        )[:3]
        data = primary.model_dump(mode="python")
        data.update(
            facts=[item.model_dump(mode="python") for item in facts],
            missing_facts=[item.model_dump(mode="python") for item in missing],
            issues=[item.model_dump(mode="python") for item in issues],
            clarification={"should_ask": bool(questions), "questions": questions},
        )
        return LegalResearchPlan.model_validate(data)


class LegalQueryPlanner:
    """Use a structured model as the primary legal-problem recognizer."""

    def __init__(
        self,
        gateway: ModelGateway,
        *,
        model: Optional[str] = None,
        reasoning_effort: Literal["low", "medium"] = "medium",
        minimum_confidence: float = 0.55,
        minimum_candidate_recall: float = 0.01,
        fallback_planner: Optional[LegacyFallbackPlanner] = None,
    ) -> None:
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be between zero and one")
        if minimum_candidate_recall < 0.0:
            raise ValueError("minimum_candidate_recall cannot be negative")
        self.gateway = gateway
        self.model = model or os.getenv("LEGAL_PLANNER_MODEL", DEFAULT_PLANNER_MODEL)
        self.reasoning_effort = reasoning_effort
        self.minimum_confidence = minimum_confidence
        self.minimum_candidate_recall = minimum_candidate_recall
        self.fallback_planner = fallback_planner or LegacyFallbackPlanner()
        self.last_outcome: Optional[PlannerOutcome] = None

    async def plan(
        self,
        question: str,
        *,
        target_date: Optional[str] = None,
        candidate_recall: Optional[float] = None,
        force_fallback: bool = False,
        locked_institutions: Optional[list[dict[str, object]]] = None,
    ) -> PlannerOutcome:
        if not question or not question.strip():
            raise ValueError("question cannot be empty")
        if force_fallback:
            return self._fallback(
                question,
                reason="forced",
                target_date=target_date,
                primary_succeeded=False,
            )

        try:
            generated = await self.gateway.generate_structured(
                response_model=LegalResearchPlan,
                input=self._planner_input(question, target_date, locked_institutions),
                system_prompt=PLANNER_SYSTEM_PROMPT,
                model=self.model,
                reasoning_effort=self.reasoning_effort,
                max_output_tokens=5000,
            )
            plan = (
                generated
                if isinstance(generated, LegalResearchPlan)
                else LegalResearchPlan.model_validate(generated)
            )
            if target_date is not None:
                plan = LegalResearchPlan.model_validate(
                    {**plan.model_dump(mode="python"), "target_date": target_date}
                )
            plan = validate_plan_grounding(plan, question)
        except (asyncio.TimeoutError, TimeoutError) as exc:
            return self._fallback(
                question,
                reason="planner_timeout",
                target_date=target_date,
                primary_error=self._safe_error(exc),
                primary_succeeded=False,
            )
        except ModelUnavailableError as exc:
            return self._fallback(
                question,
                reason="provider_unavailable",
                target_date=target_date,
                primary_error=self._safe_error(exc),
                primary_succeeded=False,
            )
        except ModelTransportError as exc:
            reason: FallbackReason = (
                "planner_timeout"
                if any(
                    marker in str(exc).lower()
                    for marker in ("timeout", "timed out", "deadline exceeded")
                )
                else "provider_unavailable"
            )
            return self._fallback(
                question,
                reason=reason,
                target_date=target_date,
                primary_error=self._safe_error(exc),
                primary_succeeded=False,
            )
        except ModelTechnicalError as exc:
            return self._fallback(
                question,
                reason="provider_unavailable",
                target_date=target_date,
                primary_error=self._safe_error(exc),
                primary_succeeded=False,
            )
        except (ModelSchemaError, ValidationError, PlannerValidationError) as exc:
            repaired = await self._repair_issue_only_plan(question, target_date=target_date)
            if repaired is None:
                return self._fallback(
                    question,
                    reason="invalid_schema",
                    target_date=target_date,
                    primary_error=self._safe_error(exc),
                    primary_succeeded=False,
                )
            plan = repaired
        except ModelGatewayError as exc:
            # Provider request failures (for example an unsupported fallback
            # model/schema combination) are technical planning failures.  They
            # may activate the traced rule planner, never a guessed answer.
            return self._fallback(
                question,
                reason="provider_unavailable",
                target_date=target_date,
                primary_error=self._safe_error(exc),
                primary_succeeded=False,
            )

        if self._plan_is_unscoped(plan):
            repaired = await self._repair_issue_only_plan(question, target_date=target_date)
            if repaired is not None and not self._plan_is_unscoped(repaired):
                plan = repaired
            else:
                return self._fallback(
                    question,
                    reason="low_confidence",
                    target_date=target_date,
                    base_plan=plan,
                    primary_error="planner returned only unscoped general tax issues",
                    primary_succeeded=True,
                )

        if plan.confidence < self.minimum_confidence:
            return self._fallback(
                question,
                reason="low_confidence",
                target_date=target_date,
                base_plan=plan,
                primary_error=(
                    f"planner confidence {plan.confidence:.3f} is below "
                    f"{self.minimum_confidence:.3f}"
                ),
                primary_succeeded=True,
            )
        if candidate_recall is not None and candidate_recall < self.minimum_candidate_recall:
            return self._fallback(
                question,
                reason="insufficient_recall",
                target_date=target_date,
                base_plan=plan,
                primary_error=(
                    f"candidate recall {candidate_recall:.3f} is below "
                    f"{self.minimum_candidate_recall:.3f}"
                ),
                primary_succeeded=True,
            )

        outcome = PlannerOutcome(
            plan=plan,
            fallback_trace=FallbackTrace(),
            planner_model=self.model,
            primary_succeeded=True,
        )
        self.last_outcome = outcome
        return outcome

    async def plan_only(self, question: str, **kwargs: object) -> LegalResearchPlan:
        """Convenience for callers which persist trace through another layer."""

        outcome = await self.plan(question, **kwargs)  # type: ignore[arg-type]
        return outcome.plan

    def fallback_for_insufficient_recall(
        self,
        question: str,
        existing_plan: LegalResearchPlan,
        *,
        target_date: Optional[str] = None,
        candidate_recall: float = 0.0,
    ) -> PlannerOutcome:
        """Augment an existing plan after retrieval, without another model call."""

        if candidate_recall >= self.minimum_candidate_recall:
            outcome = PlannerOutcome(
                plan=validate_plan_grounding(existing_plan, question),
                fallback_trace=FallbackTrace(),
                planner_model=self.model,
                primary_succeeded=True,
            )
            self.last_outcome = outcome
            return outcome
        return self._fallback(
            question,
            reason="insufficient_recall",
            target_date=target_date,
            base_plan=existing_plan,
            primary_error=(
                f"candidate recall {candidate_recall:.3f} is below "
                f"{self.minimum_candidate_recall:.3f}"
            ),
            primary_succeeded=True,
        )

    @staticmethod
    def _planner_input(
        question: str,
        target_date: Optional[str],
        locked_institutions: Optional[list[dict[str, object]]] = None,
    ) -> str:
        target = target_date or "not supplied; derive only when explicitly stated"
        locks = json.dumps(locked_institutions or [], ensure_ascii=False, sort_keys=True)
        return (
            f"Prompt version: {PLANNER_PROMPT_VERSION}\n"
            f"Authoritative target date: {target}\n"
            "The source_span offsets refer only to the exact text between "
            "<user_question> tags, excluding the tags.\n"
            f"<deterministic_institution_locks>{locks}</deterministic_institution_locks>\n"
            f"<user_question>{question}</user_question>"
        )

    async def _repair_issue_only_plan(
        self,
        question: str,
        *,
        target_date: Optional[str],
    ) -> Optional[LegalResearchPlan]:
        """Retry issue location with a smaller schema and no fact spans.

        The full planner schema is intentionally strict and can be rejected
        because of one malformed source span even when the provider correctly
        recognized the legal mechanism.  This recovery call asks only for the
        research issues and provision candidates; it cannot emit a legal
        answer and every candidate still has to pass retrieval and validation.
        """

        try:
            located = await self.gateway.generate_structured(
                response_model=IssueLocatorOutput,
                input=(
                    f"Authoritative target date: {target_date or 'not supplied'}\n"
                    f"<user_question>{question}</user_question>"
                ),
                system_prompt=ISSUE_LOCATOR_SYSTEM_PROMPT,
                model=self.model,
                reasoning_effort=self.reasoning_effort,
                max_output_tokens=4000,
            )
            output = (
                located
                if isinstance(located, IssueLocatorOutput)
                else IssueLocatorOutput.model_validate(located)
            )
            plan = LegalResearchPlan(
                user_query=question,
                intent=ResearchIntent(
                    mode="mixed_analysis",
                    needs_normative_answer=True,
                    needs_interpretations=True,
                    needs_case_law=True,
                    needs_conflict_analysis=True,
                    needs_calculations=bool(re.search(r"\d", question)),
                ),
                target_date=target_date,
                issues=output.issues,
                clarification=Clarification(),
                confidence=output.confidence,
            )
            return validate_plan_grounding(plan, question)
        except (ModelGatewayError, ValidationError, PlannerValidationError):
            return None

    @staticmethod
    def _plan_is_unscoped(plan: LegalResearchPlan) -> bool:
        if not plan.issues:
            return True
        for issue in plan.issues:
            text = " ".join(
                (
                    issue.issue_id,
                    issue.label,
                    issue.legal_mechanism,
                    *issue.possible_provision_concepts,
                    *issue.possible_legal_concepts,
                    *issue.possible_provision_hints,
                )
            ).casefold()
            generic = (
                "general tax" in text
                or "general_tax_analysis" in text
                or issue.legal_mechanism.casefold() in {"general", "analysis"}
            )
            if generic:
                return True
        return False

    def _fallback(
        self,
        question: str,
        *,
        reason: FallbackReason,
        target_date: Optional[str],
        base_plan: Optional[LegalResearchPlan] = None,
        primary_error: Optional[str] = None,
        primary_succeeded: bool,
    ) -> PlannerOutcome:
        plan, trace = self.fallback_planner.plan(
            question,
            reason=reason,
            target_date=target_date,
            base_plan=base_plan,
            primary_error=primary_error,
        )
        outcome = PlannerOutcome(
            plan=plan,
            fallback_trace=trace,
            planner_model=self.model,
            primary_succeeded=primary_succeeded,
        )
        self.last_outcome = outcome
        return outcome

    @staticmethod
    def _safe_error(exc: BaseException) -> str:
        # Provider exceptions should not contain credentials, and truncating
        # avoids copying a malformed full provider payload into trace files.
        return f"{type(exc).__name__}: {str(exc)}"[:500]
