from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from typing import Any, Iterable, Literal, Mapping, Optional, Sequence


ProvisionRelation = Literal[
    "references",
    "defines",
    "exception_to",
    "special_rule_for",
    "overrides",
    "temporal_successor",
    "neighbor",
    "transitional_rule_for",
]

RELATION_TYPES: tuple[str, ...] = (
    "references",
    "defines",
    "exception_to",
    "special_rule_for",
    "overrides",
    "temporal_successor",
    "neighbor",
    "transitional_rule_for",
)

_ARTICLE_RE = re.compile(r"^\s*art\.\s*(\d+[a-z]*)\.?\s*(.*)$", re.IGNORECASE)
_PARAGRAPH_RE = re.compile(r"^\s*§\s*(\d+[a-z]*)\.?\s*(.*)$", re.IGNORECASE)
_EXPLICIT_SECTION_RE = re.compile(r"^\s*ust\.\s*(\d+[a-z]*)\.?\s*(.*)$", re.IGNORECASE)
_EXPLICIT_POINT_RE = re.compile(r"^\s*pkt\s*(\d+[a-z]*)\.?\s*(.*)$", re.IGNORECASE)
_EXPLICIT_LETTER_RE = re.compile(r"^\s*lit\.\s*([a-z])\)?\s*(.*)$", re.IGNORECASE)
_SECTION_RE = re.compile(r"^\s*(\d+[a-z]*)\.\s+(.+)$", re.IGNORECASE)
_POINT_RE = re.compile(r"^\s*(\d+[a-z]*)\)\s*(.+)$", re.IGNORECASE)
_LETTER_RE = re.compile(r"^\s*([a-z])\)\s*(.+)$", re.IGNORECASE)
_REFERENCE_RE = re.compile(
    r"(?:(?:art\.\s*(?P<article>\d+[a-z]*))|(?:§\s*(?P<paragraph>\d+[a-z]*)))"
    r"(?:\s+ust\.\s*(?P<section>\d+[a-z]*))?"
    r"(?:\s+pkt\s*(?P<point>\d+[a-z]*))?"
    r"(?:\s+lit\.\s*(?P<letter>[a-z]))?",
    re.IGNORECASE,
)
_RELATIVE_SECTION_REFERENCE_RE = re.compile(
    r"\bust\.\s*(?P<section>\d+[a-z]*)"
    r"(?:\s+pkt\s*(?P<point>\d+[a-z]*))?"
    r"(?:\s+lit\.\s*(?P<letter>[a-z]))?",
    re.IGNORECASE,
)
_RELATIVE_POINT_REFERENCE_RE = re.compile(
    r"\bpkt\s*(?P<point>\d+[a-z]*)(?:\s+lit\.\s*(?P<letter>[a-z]))?",
    re.IGNORECASE,
)


def _parse_date(value: str | date | datetime | None) -> Optional[date]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value).strip()[:10])
    except ValueError:
        return None


@dataclass(frozen=True)
class ParsedProvisionReference:
    citation: str
    article: Optional[str] = None
    paragraph: Optional[str] = None
    section: Optional[str] = None
    point: Optional[str] = None
    letter: Optional[str] = None
    source_span_start: int = 0
    source_span_end: int = 0

    @property
    def ustep(self) -> Optional[str]:
        return self.section

    @property
    def punkt(self) -> Optional[str]:
        return self.point

    @property
    def litera(self) -> Optional[str]:
        return self.letter

    @property
    def paragraf(self) -> Optional[str]:
        return self.paragraph


@dataclass(frozen=True)
class ProvisionUnit:
    provision_id: str
    document_id: str
    version_id: str
    citation: str
    text: str
    article: Optional[str] = None
    paragraph: Optional[str] = None
    section: Optional[str] = None
    point: Optional[str] = None
    letter: Optional[str] = None
    parent_id: Optional[str] = None
    effective_from: Optional[str] = None
    effective_to: Optional[str] = None
    source_span_start: int = 0
    source_span_end: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ustep(self) -> Optional[str]:
        return self.section

    @property
    def punkt(self) -> Optional[str]:
        return self.point

    @property
    def litera(self) -> Optional[str]:
        return self.letter

    @property
    def paragraf(self) -> Optional[str]:
        return self.paragraph

    @property
    def unit_type(self) -> str:
        if self.letter:
            return "letter"
        if self.point:
            return "point"
        if self.section:
            return "section"
        if self.paragraph:
            return "paragraph"
        return "article"

    def is_effective_on(self, target_date: str | date | datetime | None) -> bool:
        target = _parse_date(target_date)
        if target is None:
            return target_date is None or target_date == ""
        start = _parse_date(self.effective_from)
        end = _parse_date(self.effective_to)
        return (start is None or start <= target) and (end is None or target <= end)


@dataclass(frozen=True)
class ProvisionEdge:
    source_id: str
    target_id: str
    relationship: ProvisionRelation
    effective_from: Optional[str] = None
    effective_to: Optional[str] = None
    evidence: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.relationship not in RELATION_TYPES:
            raise ValueError(f"Unsupported provision relationship: {self.relationship}")

    def is_effective_on(self, target_date: str | date | datetime | None) -> bool:
        target = _parse_date(target_date)
        if target is None:
            return target_date is None or target_date == ""
        start = _parse_date(self.effective_from)
        end = _parse_date(self.effective_to)
        return (start is None or start <= target) and (end is None or target <= end)


class ProvisionParser:
    """Split Polish legal text into article/section/point/letter/§ units."""

    def parse(
        self,
        text: str,
        *,
        document_id: str,
        version_id: str = "current",
        effective_from: Optional[str] = None,
        effective_to: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> tuple[ProvisionUnit, ...]:
        if not document_id.strip():
            raise ValueError("document_id cannot be empty")
        if not text.strip():
            return ()

        units: list[ProvisionUnit] = []
        current: Optional[dict[str, Any]] = None
        context: dict[str, Optional[str]] = {
            "article": None,
            "paragraph": None,
            "section": None,
            "point": None,
            "letter": None,
        }

        def finish(end: int) -> None:
            nonlocal current
            if current is None:
                return
            body = "\n".join(current.pop("body")).strip()
            current["text"] = body
            current["source_span_end"] = max(current["source_span_start"], end)
            units.append(ProvisionUnit(**current))
            current = None

        offset = 0
        for raw_line in text.splitlines(keepends=True):
            line = raw_line.rstrip("\r\n")
            marker = self._parse_marker(line, context)
            if marker is None:
                if current is not None and line.strip():
                    current["body"].append(line.strip())
                offset += len(raw_line)
                continue

            finish(offset)
            level, value, body = marker
            self._advance_context(context, level, value)
            citation = _format_citation(context)
            parent_id = self._parent_id(
                document_id=document_id,
                version_id=version_id,
                context=context,
                level=level,
            )
            current = {
                "provision_id": _provision_id(document_id, version_id, context),
                "document_id": document_id,
                "version_id": version_id,
                "citation": citation,
                "article": context["article"],
                "paragraph": context["paragraph"],
                "section": context["section"],
                "point": context["point"],
                "letter": context["letter"],
                "parent_id": parent_id,
                "effective_from": effective_from,
                "effective_to": effective_to,
                "source_span_start": offset,
                "metadata": dict(metadata or {}),
                "body": [body.strip()] if body.strip() else [],
            }
            offset += len(raw_line)
        finish(len(text))
        return tuple(units)

    parse_document = parse

    def extract_references(self, text: str) -> tuple[ParsedProvisionReference, ...]:
        references: list[ParsedProvisionReference] = []
        for match in _REFERENCE_RE.finditer(text):
            references.append(
                ParsedProvisionReference(
                    citation=match.group(0).strip(" .,;:"),
                    article=_lower(match.group("article")),
                    paragraph=_lower(match.group("paragraph")),
                    section=_lower(match.group("section")),
                    point=_lower(match.group("point")),
                    letter=_lower(match.group("letter")),
                    source_span_start=match.start(),
                    source_span_end=match.end(),
                )
            )
        return tuple(references)

    def build_graph(self, *args: Any, **kwargs: Any) -> "ProvisionGraph":
        graph = ProvisionGraph(self.parse(*args, **kwargs))
        graph.populate_inferred_edges()
        return graph

    def _parse_marker(
        self, line: str, context: Mapping[str, Optional[str]]
    ) -> Optional[tuple[str, str, str]]:
        for level, pattern in (
            ("article", _ARTICLE_RE),
            ("paragraph", _PARAGRAPH_RE),
            ("section", _EXPLICIT_SECTION_RE),
            ("point", _EXPLICIT_POINT_RE),
            ("letter", _EXPLICIT_LETTER_RE),
        ):
            match = pattern.match(line)
            if match:
                return level, match.group(1).casefold(), match.group(2)
        match = _SECTION_RE.match(line)
        if match and (context.get("article") or context.get("paragraph")):
            return "section", match.group(1).casefold(), match.group(2)
        match = _POINT_RE.match(line)
        if match and (context.get("section") or context.get("paragraph")):
            return "point", match.group(1).casefold(), match.group(2)
        match = _LETTER_RE.match(line)
        if match and context.get("point"):
            return "letter", match.group(1).casefold(), match.group(2)
        return None

    @staticmethod
    def _advance_context(
        context: dict[str, Optional[str]], level: str, value: str
    ) -> None:
        order = ("article", "paragraph", "section", "point", "letter")
        if level == "article":
            context.update(article=value, paragraph=None, section=None, point=None, letter=None)
        elif level == "paragraph":
            context.update(paragraph=value, section=None, point=None, letter=None)
        elif level == "section":
            context.update(section=value, point=None, letter=None)
        elif level == "point":
            context.update(point=value, letter=None)
        elif level == "letter":
            context["letter"] = value
        else:  # pragma: no cover - internal invariant
            raise ValueError(f"Unknown provision level {level!r}; expected one of {order}")

    @staticmethod
    def _parent_id(
        *,
        document_id: str,
        version_id: str,
        context: Mapping[str, Optional[str]],
        level: str,
    ) -> Optional[str]:
        parent_context = dict(context)
        if level == "article":
            return None
        if level == "paragraph":
            parent_context["paragraph"] = None
        elif level == "section":
            parent_context["section"] = None
        elif level == "point":
            parent_context["point"] = None
        elif level == "letter":
            parent_context["letter"] = None
        return _provision_id(document_id, version_id, parent_context)


class ProvisionGraph:
    def __init__(
        self,
        provisions: Iterable[ProvisionUnit] = (),
        edges: Iterable[ProvisionEdge] = (),
    ) -> None:
        self._provisions: dict[str, ProvisionUnit] = {}
        self._edges: list[ProvisionEdge] = []
        for provision in provisions:
            self.add_provision(provision)
        for edge in edges:
            self.add_edge(edge)

    @property
    def provisions(self) -> tuple[ProvisionUnit, ...]:
        return tuple(self._provisions.values())

    @property
    def nodes(self) -> tuple[ProvisionUnit, ...]:
        return self.provisions

    @property
    def edges(self) -> tuple[ProvisionEdge, ...]:
        return tuple(self._edges)

    def add_provision(self, provision: ProvisionUnit) -> None:
        self._provisions[provision.provision_id] = provision

    add_unit = add_provision

    def add_edge(
        self,
        edge: ProvisionEdge | str,
        target_id: Optional[str] = None,
        relationship: Optional[ProvisionRelation] = None,
        **kwargs: Any,
    ) -> None:
        if isinstance(edge, ProvisionEdge):
            candidate = edge
        else:
            if target_id is None or relationship is None:
                raise ValueError("target_id and relationship are required")
            candidate = ProvisionEdge(edge, target_id, relationship, **kwargs)
        if candidate.source_id not in self._provisions:
            raise KeyError(f"Unknown source provision: {candidate.source_id}")
        if candidate.target_id not in self._provisions:
            raise KeyError(f"Unknown target provision: {candidate.target_id}")
        key = (candidate.source_id, candidate.target_id, candidate.relationship)
        if not any((item.source_id, item.target_id, item.relationship) == key for item in self._edges):
            self._edges.append(candidate)

    def get(
        self, provision_id: str, target_date: str | date | datetime | None = None
    ) -> Optional[ProvisionUnit]:
        provision = self._provisions.get(provision_id)
        if provision is None or not provision.is_effective_on(target_date):
            return None
        return provision

    def active_provisions(
        self, target_date: str | date | datetime | None
    ) -> tuple[ProvisionUnit, ...]:
        return tuple(
            item for item in self._provisions.values() if item.is_effective_on(target_date)
        )

    def filter_for_date(
        self, target_date: str | date | datetime
    ) -> "ProvisionGraph":
        provisions = self.active_provisions(target_date)
        ids = {item.provision_id for item in provisions}
        edges = (
            edge
            for edge in self._edges
            if edge.source_id in ids
            and edge.target_id in ids
            and edge.is_effective_on(target_date)
        )
        return ProvisionGraph(provisions, edges)

    def related(
        self,
        provision_id: str,
        *,
        relationships: Optional[Iterable[ProvisionRelation]] = None,
        target_date: str | date | datetime | None = None,
        direction: Literal["outgoing", "incoming", "both"] = "outgoing",
        max_depth: int = 1,
    ) -> tuple[ProvisionUnit, ...]:
        if max_depth < 1 or self.get(provision_id, target_date) is None:
            return ()
        allowed = set(relationships or RELATION_TYPES)
        seen = {provision_id}
        frontier = [provision_id]
        result: list[ProvisionUnit] = []
        for _ in range(max_depth):
            next_frontier: list[str] = []
            for current in frontier:
                for edge in self._edges:
                    if edge.relationship not in allowed or not edge.is_effective_on(target_date):
                        continue
                    target: Optional[str] = None
                    if direction in {"outgoing", "both"} and edge.source_id == current:
                        target = edge.target_id
                    elif direction in {"incoming", "both"} and edge.target_id == current:
                        target = edge.source_id
                    if target is None or target in seen:
                        continue
                    provision = self.get(target, target_date)
                    if provision is None:
                        continue
                    seen.add(target)
                    result.append(provision)
                    next_frontier.append(target)
            frontier = next_frontier
            if not frontier:
                break
        return tuple(result)

    def resolve_dependencies(
        self,
        provision_ids: Iterable[str],
        *,
        target_date: str | date | datetime | None,
        max_depth: int = 2,
    ) -> tuple[ProvisionUnit, ...]:
        relationships: tuple[ProvisionRelation, ...] = (
            "references",
            "defines",
            "exception_to",
            "special_rule_for",
            "overrides",
            "temporal_successor",
            "neighbor",
            "transitional_rule_for",
        )
        result: list[ProvisionUnit] = []
        seen: set[str] = set()
        for provision_id in provision_ids:
            root = self.get(provision_id, target_date)
            if root is not None and root.provision_id not in seen:
                seen.add(root.provision_id)
                result.append(root)
            for related in self.related(
                provision_id,
                relationships=relationships,
                target_date=target_date,
                direction="both",
                max_depth=max_depth,
            ):
                if related.provision_id not in seen:
                    seen.add(related.provision_id)
                    result.append(related)
        return tuple(result)

    def populate_inferred_edges(self) -> None:
        self._add_neighbor_edges()
        self._add_temporal_successor_edges()
        parser = ProvisionParser()
        for source in tuple(self._provisions.values()):
            references = list(parser.extract_references(source.text))
            occupied = [
                (reference.source_span_start, reference.source_span_end)
                for reference in references
            ]
            for pattern in (
                _RELATIVE_SECTION_REFERENCE_RE,
                _RELATIVE_POINT_REFERENCE_RE,
            ):
                for match in pattern.finditer(source.text):
                    if any(start <= match.start() < end for start, end in occupied):
                        continue
                    references.append(
                        ParsedProvisionReference(
                            citation=match.group(0).strip(" .,;:"),
                            article=source.article,
                            paragraph=source.paragraph,
                            section=(
                                _lower(match.groupdict().get("section"))
                                or source.section
                            ),
                            point=_lower(match.groupdict().get("point")),
                            letter=_lower(match.groupdict().get("letter")),
                            source_span_start=match.start(),
                            source_span_end=match.end(),
                        )
                    )
            for reference in references:
                target = self.find_reference(source.document_id, source.version_id, reference)
                if target is None or target.provision_id == source.provision_id:
                    continue
                relationship = _infer_relationship(source.text, reference)
                self.add_edge(
                    ProvisionEdge(
                        source.provision_id,
                        target.provision_id,
                        relationship,
                        effective_from=source.effective_from,
                        effective_to=source.effective_to,
                        evidence=reference.citation,
                    )
                )

    def find_reference(
        self,
        document_id: str,
        version_id: str,
        reference: ParsedProvisionReference,
    ) -> Optional[ProvisionUnit]:
        candidates = [
            item
            for item in self._provisions.values()
            if item.document_id == document_id
            and item.version_id == version_id
            and item.article == reference.article
            and item.paragraph == reference.paragraph
            and (reference.section is None or item.section == reference.section)
            and (reference.point is None or item.point == reference.point)
            and (reference.letter is None or item.letter == reference.letter)
        ]
        if not candidates:
            return None
        # Prefer the exact referenced editorial unit over one of its children.
        # E.g. `art. 21 ust. 1` must not silently resolve to pkt 1 lit. a.
        candidates.sort(key=lambda item: _specificity(item))
        return candidates[0]

    def _add_neighbor_edges(self) -> None:
        groups: dict[tuple[str, str, Optional[str]], list[ProvisionUnit]] = {}
        for item in self._provisions.values():
            groups.setdefault((item.document_id, item.version_id, item.parent_id), []).append(item)
        for siblings in groups.values():
            siblings.sort(key=lambda item: item.source_span_start)
            for left, right in zip(siblings, siblings[1:]):
                self.add_edge(ProvisionEdge(left.provision_id, right.provision_id, "neighbor"))
                self.add_edge(ProvisionEdge(right.provision_id, left.provision_id, "neighbor"))

    def _add_temporal_successor_edges(self) -> None:
        groups: dict[tuple[str, str], list[ProvisionUnit]] = {}
        for item in self._provisions.values():
            groups.setdefault((item.document_id, item.citation.casefold()), []).append(item)
        for versions in groups.values():
            versions.sort(
                key=lambda item: (
                    _parse_date(item.effective_from) or date.min,
                    item.version_id,
                )
            )
            for previous, successor in zip(versions, versions[1:]):
                if previous.version_id == successor.version_id:
                    continue
                self.add_edge(
                    ProvisionEdge(
                        previous.provision_id,
                        successor.provision_id,
                        "temporal_successor",
                    )
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provisions": [dict(item.__dict__) for item in self.provisions],
            "edges": [dict(item.__dict__) for item in self.edges],
        }

    def to_schema(self) -> Any:
        """Convert the runtime graph to the pipeline's strict Pydantic schema."""

        from .schemas import (
            DocumentSourceSpan,
            ProvisionGraph as ProvisionGraphSchema,
            ProvisionGraphEdge,
            ProvisionReference,
        )

        provisions = []
        for item in self.provisions:
            span = None
            if item.source_span_end > item.source_span_start:
                span = DocumentSourceSpan(
                    start=item.source_span_start,
                    end=item.source_span_end,
                    source_id="legal_document",
                    document_id=item.document_id,
                )
            provisions.append(
                ProvisionReference(
                    provision_id=item.provision_id,
                    document_id=item.document_id,
                    version_id=item.version_id,
                    citation=item.citation,
                    article=item.article,
                    paragraph=item.section or item.paragraph,
                    point=item.point,
                    letter=item.letter,
                    effective_from=item.effective_from,
                    effective_to=item.effective_to,
                    status="unknown",
                    text=item.text or None,
                    source_span=span,
                )
            )
        return ProvisionGraphSchema(
            provisions=provisions,
            edges=[
                ProvisionGraphEdge(
                    source_provision_id=edge.source_id,
                    target_provision_id=edge.target_id,
                    relationship=edge.relationship,
                    verified=bool(edge.metadata.get("verified", False)),
                )
                for edge in self.edges
            ],
        )


def _lower(value: Optional[str]) -> Optional[str]:
    return value.casefold() if value else None


def _format_citation(context: Mapping[str, Optional[str]]) -> str:
    parts: list[str] = []
    if context.get("article"):
        parts.append(f"art. {context['article']}")
    if context.get("paragraph"):
        parts.append(f"§ {context['paragraph']}")
    if context.get("section"):
        parts.append(f"ust. {context['section']}")
    if context.get("point"):
        parts.append(f"pkt {context['point']}")
    if context.get("letter"):
        parts.append(f"lit. {context['letter']}")
    return " ".join(parts)


def _provision_id(
    document_id: str, version_id: str, context: Mapping[str, Optional[str]]
) -> str:
    path = _format_citation(context) or "document"
    digest = hashlib.sha256(f"{document_id}\0{version_id}\0{path}".encode("utf-8")).hexdigest()[:16]
    return f"{document_id}:{version_id}:{digest}"


def _specificity(item: ProvisionUnit) -> int:
    return sum(bool(value) for value in (item.article, item.paragraph, item.section, item.point, item.letter))


def _infer_relationship(
    text: str, reference: ParsedProvisionReference
) -> ProvisionRelation:
    start = max(0, reference.source_span_start - 120)
    end = min(len(text), reference.source_span_end + 80)
    context = text[start:end].casefold()
    if re.search(r"\b(przepis(?:y)? przejściow|do spraw wszczętych|do okresów rozpoczętych)\b", context):
        return "transitional_rule_for"
    if re.search(r"\b(z wyjątkiem|nie stosuje się|wyłącza się|chyba że)\b", context):
        return "exception_to"
    if re.search(r"\b(z zastrzeżeniem|przepis szczególny|na zasadach określonych)\b", context):
        return "special_rule_for"
    if re.search(r"\b(stosuje się zamiast|ma pierwszeństwo|bez względu na)\b", context):
        return "overrides"
    if re.search(r"\b(w rozumieniu|rozumie się przez|oznacza)\b", context):
        return "defines"
    return "references"
