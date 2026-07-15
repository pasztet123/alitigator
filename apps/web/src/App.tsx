import { useEffect, useRef, useState } from 'react'
import type { FormEvent, ReactNode } from 'react'
import type { Session } from '@supabase/supabase-js'
import './App.css'
import logoUrl from './assets/alitigator-logo.png'
import { isSupabaseConfigured, supabaseBrowserClient } from './lib/supabase'

type Role = 'user' | 'assistant'
type ChatMode = 'demo' | 'live'
type RetrievalScope = 'statutes_only' | 'statutes_and_interpretations' | 'full_argumentation'

type Message = {
  id: string
  role: Role
  content: string
  created_at?: string
  feedback_rating?: number | null
  feedback_comment?: string | null
  feedback_created_at?: string | null
  structured_reply?: StructuredReply
}

type StructuredReplySection = {
  key: string
  title: string
  content: string
}

type StructuredReply = {
  opening_statute?: string | null
  sections: StructuredReplySection[]
}

type ChatResponse = {
  reply: string
  mode: ChatMode
  model: string
  redactions: string[]
  analysis_trace?: Record<string, unknown>
  chat_id?: string
  assistant_message_id?: string | null
  structured_reply?: StructuredReply | null
}

type ChatMessageFeedbackRequest = {
  rating: number
  comment?: string | null
}

type RetrievalPreferences = {
  include_interpretations: boolean
  include_judgments: boolean
}

type IntentHintAnswer = {
  question: string
  option_id: string
  option_label: string
}

type PromptHintOption = {
  id: string
  label: string
}

type PromptHint = {
  id: string
  question: string
  options: PromptHintOption[]
}

type PromptHintsResponse = {
  hints: PromptHint[]
  model: string
  mode: 'live' | 'fallback'
}

type ModelsResponse = {
  default_model: string
  models: string[]
}

type HealthResponse = {
  status: string
  version: string
  llm_configured: boolean
  llm_provider: string
  supabase_configured: boolean
  rag_index_configured: boolean
  chat_storage_available: boolean
  auth_configured: boolean
  stripe_configured: boolean
}

type ThreadSummary = {
  id: string
  title: string
  archived: boolean
  updated_at: string
  created_at: string
  last_message_preview: string
}

type ThreadsResponse = {
  active: ThreadSummary[]
  archived: ThreadSummary[]
}

type ThreadDetail = ThreadSummary & {
  messages: Message[]
}

type LocalThreadMessages = Record<string, Message[]>
type FeedbackDrafts = Record<string, string>

type Profile = {
  id: string
  email?: string | null
  full_name?: string | null
  law_firm?: string | null
  is_admin?: boolean
  stripe_customer_id?: string | null
  created_at?: string | null
  updated_at?: string | null
}

type CreditPack = {
  id: string
  name: string
  credit_amount: number
  price_gross: number
  currency: string
  description: string
}

type AccountResponse = {
  user_id: string
  email?: string | null
  profile: Profile
  is_admin: boolean
  credit_balance: number
  credit_cost_per_query: number
  credit_unit_price_gross: number
  credit_currency: string
  stripe_configured: boolean
  credit_packs: CreditPack[]
}

type AdminGrantCreditsResponse = {
  user_id: string
  email?: string | null
  full_name?: string | null
  credit_balance: number
}

type AdminUserSummary = {
  user_id: string
  email?: string | null
  full_name?: string | null
  law_firm?: string | null
  is_admin: boolean
  credit_balance: number
  created_at?: string | null
}

type AdminUsersResponse = {
  users: AdminUserSummary[]
}

type CheckoutSessionStatusResponse = {
  checkout_session_id: string
  payment_status: string
  status?: string | null
  credited: boolean
}

type AuthMode = 'signin' | 'signup'

type AssistantSection = {
  key?: string
  title: string
  content: string
}

type ParsedAssistantMessage = {
  intro: string
  sections: AssistantSection[]
}

const modelLabels: Record<string, string> = {
  'gpt-5.6-terra': 'GPT-5.6 Terra',
  'gpt-5.6-luna': 'GPT-5.6 Luna',
  'claude-opus-4-8': 'Claude Opus 4.8',
  'claude-sonnet-4-6': 'Claude Sonnet 4.6',
  'claude-haiku-4-5-20251001': 'Claude Haiku 4.5',
}

const configuredApiBaseUrl = (import.meta.env.VITE_API_BASE_URL ?? '').trim().replace(/\/+$/, '')
const API_BASE_URL = configuredApiBaseUrl || (import.meta.env.DEV ? 'http://localhost:8000' : '')
const LOCAL_THREAD_PREFIX = 'local-thread-'
const HINT_DEBOUNCE_MS = 900
const MIN_DRAFT_LENGTH_FOR_HINTS = 24
const ACTIVE_HINT_COUNT = 3
const MAX_HINT_QUESTION_COUNT = 5
const APP_VERSION = '2.0.24'
const ASSISTANT_SECTION_TITLES = [
  'Teza',
  'Analiza',
  'Źródła',
  'Ryzyka i luki',
  'Źródła zwrócone przez retrieval',
  'Źródła użyte przez retrieval',
] as const

function isLocalThreadId(chatId: string | null | undefined): chatId is string {
  return Boolean(chatId?.startsWith(LOCAL_THREAD_PREFIX))
}

function buildThreadTitle(content: string) {
  const compact = content.trim().replace(/\s+/g, ' ')
  return compact.slice(0, 54).replace(/[\s,.;:-]+$/, '') || 'Nowy wątek'
}

function buildThreadPreview(content: string) {
  return content.trim().replace(/\s+/g, ' ').slice(0, 120)
}

function buildIntentHintsPayload(
  intentHintAnswers: Record<string, IntentHintAnswer>,
): IntentHintAnswer[] {
  return Object.values(intentHintAnswers)
}

function buildRetrievalPreferences(scope: RetrievalScope): RetrievalPreferences {
  if (scope === 'statutes_only') {
    return { include_interpretations: false, include_judgments: false }
  }
  if (scope === 'statutes_and_interpretations') {
    return { include_interpretations: true, include_judgments: false }
  }
  return { include_interpretations: true, include_judgments: true }
}

function slugifyHintOption(label: string) {
  const normalized = label
    .toLowerCase()
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')

  return normalized.slice(0, 48) || 'opcja'
}

function normalizeHintOptions(hint: Partial<PromptHint>) {
  if (Array.isArray(hint.options) && hint.options.length >= 2) {
    return hint.options
      .map((option) => ({
        id: String(option?.id ?? '').trim() || slugifyHintOption(String(option?.label ?? '')),
        label: String(option?.label ?? '').trim(),
      }))
      .filter((option) => option.label)
  }

  return []
}

function normalizePromptHintsResponse(payload: PromptHintsResponse | { hints?: unknown; model?: unknown; mode?: unknown }): PromptHintsResponse {
  const rawHints = Array.isArray(payload.hints) ? payload.hints : []
  const seenIds = new Map<string, number>()
  const seenQuestions = new Set<string>()
  const hints = rawHints
    .map((hint, index) => {
      const question = String((hint as PromptHint)?.question ?? '').trim()
      if (!question) {
        return null
      }
      const normalizedQuestion = question.toLowerCase()
      if (seenQuestions.has(normalizedQuestion)) {
        return null
      }
      seenQuestions.add(normalizedQuestion)

      const options = normalizeHintOptions(hint as Partial<PromptHint>)
      const baseId = String((hint as PromptHint)?.id ?? '').trim() || `hint-${index}-${slugifyHintOption(question)}`
      const duplicateCount = seenIds.get(baseId) ?? 0
      seenIds.set(baseId, duplicateCount + 1)
      return {
        id: duplicateCount === 0 ? baseId : `${baseId}-${duplicateCount + 1}`,
        question,
        options,
      }
    })
    .filter((hint): hint is PromptHint => Boolean(hint))

  return {
    hints,
    model: String(payload.model ?? 'fallback'),
    mode: payload.mode === 'live' ? 'live' : 'fallback',
  }
}

function escapeRegExp(value: string) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

function parseAssistantMessage(content: string): ParsedAssistantMessage {
  const normalized = content.replace(/\r\n/g, '\n').trim()
  if (!normalized) {
    return { intro: '', sections: [] }
  }

  const headingsPattern = ASSISTANT_SECTION_TITLES.map(escapeRegExp).join('|')
  const sectionRegex = new RegExp(`(^|\\n\\n)(${headingsPattern})\\n`, 'g')
  const matches = Array.from(normalized.matchAll(sectionRegex))

  if (matches.length === 0) {
    return { intro: normalized, sections: [] }
  }

  const firstMatch = matches[0]
  const intro = normalized.slice(0, firstMatch.index ?? 0).trim()
  const sections: AssistantSection[] = matches.map((match, index) => {
    const title = match[2]
    const contentStart = (match.index ?? 0) + match[0].length
    const nextStart = index + 1 < matches.length ? (matches[index + 1].index ?? normalized.length) : normalized.length
    return {
      title,
      content: normalized.slice(contentStart, nextStart).trim(),
    }
  })

  return { intro, sections }
}

function isSectionOpenByDefault(title: string) {
  return title === 'Teza' || title === 'Analiza'
}

function renderInlineRichText(text: string): ReactNode[] {
  const segments = text.split(/(\*\*[^*]+\*\*)/g)
  return segments.map((segment, index) => {
    if (segment.startsWith('**') && segment.endsWith('**') && segment.length > 4) {
      return <strong key={`strong-${index}`}>{segment.slice(2, -2)}</strong>
    }

    return <span key={`text-${index}`}>{segment}</span>
  })
}

function renderInlineHeading(level: number, content: string, key: string) {
  const richContent = renderInlineRichText(content)

  if (level <= 3) {
    return <h3 key={key} className="assistant-inline-heading">{richContent}</h3>
  }
  if (level === 4) {
    return <h4 key={key} className="assistant-inline-heading">{richContent}</h4>
  }
  return <h5 key={key} className="assistant-inline-heading">{richContent}</h5>
}

function renderRichText(content: string, baseKey: string) {
  const blocks = content
    .split(/\n{2,}/)
    .map((block) => block.trim())
    .filter(Boolean)

  return blocks.map((block, blockIndex) => {
    const lines = block.split('\n').map((line) => line.trim()).filter(Boolean)
    if (!lines.length) {
      return null
    }

    const headingMatch = lines[0].match(/^(#{1,4})\s+(.+)$/)
    if (headingMatch) {
      const headingLevel = Math.min(headingMatch[1].length + 2, 6)
      const bodyLines = lines.slice(1)
      return (
        <section key={`${baseKey}-heading-${blockIndex}`} className="assistant-rich-block">
          {renderInlineHeading(headingLevel, headingMatch[2], `${baseKey}-heading-title-${blockIndex}`)}
          {bodyLines.length > 0 ? (
            <p className="message-content">
              {renderInlineRichText(bodyLines.join(' '))}
            </p>
          ) : null}
        </section>
      )
    }

    if (lines.every((line) => /^[-*]\s+/.test(line))) {
      return (
        <ul key={`${baseKey}-list-${blockIndex}`} className="assistant-rich-list">
          {lines.map((line, lineIndex) => (
            <li key={`${baseKey}-item-${blockIndex}-${lineIndex}`}>
              {renderInlineRichText(line.replace(/^[-*]\s+/, ''))}
            </li>
          ))}
        </ul>
      )
    }

    return (
      <p key={`${baseKey}-paragraph-${blockIndex}`} className="message-content">
        {renderInlineRichText(lines.join(' '))}
      </p>
    )
  })
}

function AssistantMessageBody({
  content,
  structuredReply,
}: {
  content: string
  structuredReply?: StructuredReply
}) {
  const parsed = structuredReply
    ? {
        intro: structuredReply.opening_statute?.trim() ?? '',
        sections: structuredReply.sections,
      }
    : parseAssistantMessage(content)

  if (!parsed.sections.length) {
    return <div className="assistant-rich-text">{renderRichText(content, 'assistant-message')}</div>
  }

  return (
    <div className="assistant-message-body">
      {parsed.intro ? (
        <section className="assistant-statute-block" aria-label="Przepis otwierający odpowiedź">
          <p className="assistant-statute-label">Przepis</p>
          <div className="assistant-rich-text">{renderRichText(parsed.intro, 'assistant-intro')}</div>
        </section>
      ) : null}

      <div className="assistant-sections">
        {parsed.sections.map((section) => (
          <details
            key={section.key ?? `${section.title}-${section.content.slice(0, 24)}`}
            className="assistant-section"
            open={isSectionOpenByDefault(section.title)}
          >
            <summary className="assistant-section-summary">{section.title}</summary>
            <div className="assistant-section-panel">
              <div className="assistant-rich-text">
                {renderRichText(section.content, section.key ?? section.title)}
              </div>
            </div>
          </details>
        ))}
      </div>
    </div>
  )
}

function createOptimisticThread(initialMessage: string): ThreadSummary {
  const now = new Date().toISOString()

  return {
    id: `${LOCAL_THREAD_PREFIX}${crypto.randomUUID()}`,
    title: buildThreadTitle(initialMessage),
    archived: false,
    updated_at: now,
    created_at: now,
    last_message_preview: buildThreadPreview(initialMessage),
  }
}

function formatThreadTimestamp(value: string) {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) {
    return ''
  }

  const now = new Date()
  const isSameDay = date.toDateString() === now.toDateString()

  return new Intl.DateTimeFormat('pl-PL', {
    day: isSameDay ? undefined : '2-digit',
    month: isSameDay ? undefined : '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date)
}

function formatTokenCount(value: number) {
  return new Intl.NumberFormat('pl-PL').format(value)
}

function StatusGlyph({ kind }: { kind: 'credits' | 'model' | 'shield' | 'account' }) {
  if (kind === 'credits') {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M12 3 4 7v5c0 5.1 3.4 9.8 8 11 4.6-1.2 8-5.9 8-11V7l-8-4Z" />
        <path d="M9 12h6" />
        <path d="M12 9v6" />
      </svg>
    )
  }

  if (kind === 'model') {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <rect x="4" y="5" width="16" height="14" rx="4" />
        <path d="M9 10h6" />
        <path d="M9 14h3" />
      </svg>
    )
  }

  if (kind === 'account') {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8Z" />
        <path d="M5 20a7 7 0 0 1 14 0" />
      </svg>
    )
  }

  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 3 5 6v6c0 4.4 2.8 8.5 7 9.8 4.2-1.3 7-5.4 7-9.8V6l-7-3Z" />
      <path d="m9.5 12 1.7 1.7 3.5-3.7" />
    </svg>
  )
}

function SidebarActionIcon({ kind }: { kind: 'new' | 'archive' | 'restore' | 'logout' }) {
  if (kind === 'new') {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M12 5v14" />
        <path d="M5 12h14" />
      </svg>
    )
  }

  if (kind === 'restore') {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M8 7H4v4" />
        <path d="M4 11a8 8 0 1 0 2.3-5.7L4 7" />
      </svg>
    )
  }

  if (kind === 'logout') {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M10 17H6a2 2 0 0 1-2-2V9a2 2 0 0 1 2-2h4" />
        <path d="M14 8l6 4-6 4" />
        <path d="M20 12H9" />
      </svg>
    )
  }

  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 7h16" />
      <path d="M9 7V5h6v2" />
      <path d="M7 7l1 11h8l1-11" />
    </svg>
  )
}

function SendIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M21 3 10 14" />
      <path d="m21 3-7 18-4-7-7-4 18-7Z" />
    </svg>
  )
}

async function readErrorResponse(response: Response) {
  try {
    const payload = await response.json() as { detail?: string }
    if (payload.detail) {
      return payload.detail
    }
  } catch {
    return `Backend zwrócił ${response.status}`
  }

  return `Backend zwrócił ${response.status}`
}

function buildApiUrl(path: string) {
  return `${API_BASE_URL}${path}`
}

function describeApiNetworkError(error: unknown) {
  const originalMessage = error instanceof Error ? error.message : ''
  const apiTarget = API_BASE_URL || 'ten sam host co frontend'
  const hint = configuredApiBaseUrl
    ? `Sprawdź, czy backend ${apiTarget} odpowiada i dopuszcza origin frontendu.`
    : 'W produkcji nie ustawiono VITE_API_BASE_URL, więc frontend próbuje używać tego samego hosta co aplikacja.'

  if (/load failed|failed to fetch|network|cors/i.test(originalMessage)) {
    return `Nie udało się połączyć z backendem. ${hint}`
  }

  return originalMessage || `Nie udało się połączyć z backendem. ${hint}`
}

function describeSupabaseAuthError(error: unknown) {
  const originalMessage = error instanceof Error ? error.message : ''
  if (/load failed|failed to fetch|network|cors/i.test(originalMessage)) {
    return 'Nie udało się odświeżyć sesji Supabase. Zaloguj się ponownie; jeśli błąd wraca, trzeba sprawdzić konfigurację Supabase URL / publishable key oraz dozwolone originy projektu.'
  }
  return originalMessage || 'Nie udało się odświeżyć sesji Supabase.'
}

async function appFetch(path: string, init?: RequestInit) {
  try {
    return await fetch(buildApiUrl(path), init)
  } catch (fetchError) {
    throw new Error(describeApiNetworkError(fetchError))
  }
}

async function apiFetch(path: string, session: Session, init?: RequestInit) {
  const headers = new Headers(init?.headers)
  headers.set('Authorization', `Bearer ${session.access_token}`)
  if (init?.body && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json')
  }

  return appFetch(path, {
    ...init,
    headers,
  })
}

function App() {
  const [session, setSession] = useState<Session | null>(null)
  const [authMode, setAuthMode] = useState<AuthMode>('signin')
  const [authEmail, setAuthEmail] = useState('')
  const [authPassword, setAuthPassword] = useState('')
  const [authFullName, setAuthFullName] = useState('')
  const [authLoading, setAuthLoading] = useState(true)
  const [authSubmitting, setAuthSubmitting] = useState(false)
  const [authError, setAuthError] = useState('')
  const [authInfo, setAuthInfo] = useState('')

  const [messages, setMessages] = useState<Message[]>([])
  const [draft, setDraft] = useState('')
  const [error, setError] = useState('')
  const [isSending, setIsSending] = useState(false)
  const [isBootstrapping, setIsBootstrapping] = useState(true)
  const [isThreadLoading, setIsThreadLoading] = useState(false)
  const [isSidebarBusy, setIsSidebarBusy] = useState(false)
  const [activeThreads, setActiveThreads] = useState<ThreadSummary[]>([])
  const [archivedThreads, setArchivedThreads] = useState<ThreadSummary[]>([])
  const [selectedChatId, setSelectedChatId] = useState<string | null>(null)
  const [showArchived, setShowArchived] = useState(false)
  const [localThreadMessages, setLocalThreadMessages] = useState<LocalThreadMessages>({})
  const [availableModels, setAvailableModels] = useState<string[]>(['gpt-5.6-terra'])
  const [selectedModel, setSelectedModel] = useState('gpt-5.6-terra')
  const [lastRedactions, setLastRedactions] = useState<string[]>([])
  const [chatStorageAvailable, setChatStorageAvailable] = useState(false)
  const [backendVersion, setBackendVersion] = useState<string | null>(null)
  const [promptHints, setPromptHints] = useState<PromptHint[]>([])
  const [intentHintAnswers, setIntentHintAnswers] = useState<Record<string, IntentHintAnswer>>({})
  const [isHintsLoading, setIsHintsLoading] = useState(false)
  const [hintMode, setHintMode] = useState<'live' | 'fallback'>('fallback')
  const retrievalScope: RetrievalScope = 'full_argumentation'

  const [account, setAccount] = useState<AccountResponse | null>(null)
  const [adminGrantDraft, setAdminGrantDraft] = useState({ user_email: '', credit_amount: '1', reason: '' })
  const [isAdminGrantSubmitting, setIsAdminGrantSubmitting] = useState(false)
  const [adminGrantInfo, setAdminGrantInfo] = useState('')
  const [adminUsers, setAdminUsers] = useState<AdminUserSummary[]>([])
  const [isAdminUsersLoading, setIsAdminUsersLoading] = useState(false)
  const [adminUsersError, setAdminUsersError] = useState('')
  const [isAdminPanelOpen, setIsAdminPanelOpen] = useState(false)
  const [feedbackDrafts, setFeedbackDrafts] = useState<FeedbackDrafts>({})
  const [feedbackSavingMessageId, setFeedbackSavingMessageId] = useState<string | null>(null)
  const activeWorkspaceRef = useRef({
    draft: '',
    hasMessages: false,
    isSending: false,
    selectedChatId: null as string | null,
  })
  const activeHintRequestControllerRef = useRef<AbortController | null>(null)

  useEffect(() => {
    activeWorkspaceRef.current = {
      draft,
      hasMessages: messages.length > 0,
      isSending,
      selectedChatId,
    }
  }, [draft, isSending, messages.length, selectedChatId])

  async function refreshAdminUsers(activeSession: Session) {
    setAdminUsersError('')
    const response = await apiFetch('/api/admin/users', activeSession)
    if (!response.ok) {
      throw new Error(await readErrorResponse(response))
    }

    const payload = await response.json() as AdminUsersResponse
    setAdminUsers(payload.users)
    return payload.users
  }

async function fetchPromptHints({
  draftText,
    answeredHints,
    excludedQuestions,
    maxHints,
    signal,
  }: {
    draftText: string
    answeredHints: IntentHintAnswer[]
  excludedQuestions: string[]
  maxHints: number
  signal?: AbortSignal
}) {
  const normalizedDraft = draftText.trim()
  if (!normalizedDraft || maxHints < 1) {
    return { hints: [], model: 'fallback', mode: 'fallback' as const }
  }

  const response = await appFetch('/api/chat/hints', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    signal,
    body: JSON.stringify({
        draft: normalizedDraft,
        intent_hints: answeredHints,
        excluded_questions: excludedQuestions.filter((question) => question.trim()),
        max_hints: maxHints,
      }),
    })

    if (!response.ok) {
      throw new Error(await readErrorResponse(response))
    }

    return normalizePromptHintsResponse(await response.json())
  }

  async function refreshAccount(activeSession: Session) {
    const response = await apiFetch('/api/account', activeSession)
    if (!response.ok) {
      throw new Error(await readErrorResponse(response))
    }

    const payload = await response.json() as AccountResponse
    setAccount(payload)
    return payload
  }

  async function reconcileCheckoutReturn(activeSession: Session) {
    const currentUrl = new URL(window.location.href)
    const checkoutResult = currentUrl.searchParams.get('checkout')
    const sessionId = currentUrl.searchParams.get('session_id')

    if (checkoutResult !== 'success' || !sessionId) {
      if (checkoutResult === 'cancel') {
        setAuthInfo('Zakup został anulowany.')
      }
      return
    }

    try {
      const response = await apiFetch(`/api/billing/checkout-session/${encodeURIComponent(sessionId)}`, activeSession)
      if (!response.ok) {
        throw new Error(await readErrorResponse(response))
      }

      const payload = await response.json() as CheckoutSessionStatusResponse
      await refreshAccount(activeSession)

      if (payload.credited) {
        setAuthInfo('Płatność zakończona. Kredyty są już dodane do konta.')
      } else {
        setAuthInfo('Płatność została przyjęta. Odświeżam saldo konta.')
      }
    } catch (checkoutError) {
      setError(checkoutError instanceof Error ? checkoutError.message : 'Nie udało się potwierdzić płatności Stripe.')
    } finally {
      currentUrl.searchParams.delete('checkout')
      currentUrl.searchParams.delete('session_id')
      window.history.replaceState({}, document.title, `${currentUrl.pathname}${currentUrl.search}${currentUrl.hash}`)
    }
  }

  function resetWorkspaceState() {
    setMessages([])
    setDraft('')
    setError('')
    setIsSending(false)
    setActiveThreads([])
    setArchivedThreads([])
    setSelectedChatId(null)
    setShowArchived(false)
    setLocalThreadMessages({})
    setPromptHints([])
    setIntentHintAnswers({})
    setIsHintsLoading(false)
    setHintMode('fallback')
    setAccount(null)
    setAdminGrantDraft({ user_email: '', credit_amount: '1', reason: '' })
    setAdminGrantInfo('')
    setAdminUsers([])
    setAdminUsersError('')
    setIsAdminPanelOpen(false)
    setLastRedactions([])
    setFeedbackDrafts({})
    setFeedbackSavingMessageId(null)
  }

  useEffect(() => {
    if (!supabaseBrowserClient) {
      setAuthLoading(false)
      return
    }

    const authClient = supabaseBrowserClient
    let isMounted = true

    async function restoreSession() {
      try {
        const { data, error: refreshError } = await authClient.auth.refreshSession()
        if (!isMounted) {
          return
        }
        if (refreshError && refreshError.message) {
          setAuthError(refreshError.message)
        }
        setSession(data.session ?? null)
      } catch (refreshError) {
        if (!isMounted) {
          return
        }
        setSession(null)
        setAuthError(describeSupabaseAuthError(refreshError))
        void authClient.auth.signOut({ scope: 'local' }).catch(() => undefined)
      } finally {
        if (isMounted) {
          setAuthLoading(false)
        }
      }
    }

    void restoreSession()

    const { data } = authClient.auth.onAuthStateChange((_event, nextSession) => {
      setSession(nextSession)
      setAuthLoading(false)
      setAuthError('')
      setAuthInfo('')
      if (!nextSession) {
        resetWorkspaceState()
      }
    })

    return () => {
      isMounted = false
      data.subscription.unsubscribe()
    }
  }, [])

  useEffect(() => {
    if (!session) {
      setIsBootstrapping(false)
      return
    }

    const activeSession = session
    let isCancelled = false

    async function bootstrap() {
      setIsBootstrapping(true)
      setError('')

      try {
        const [healthResponse, modelsResponse] = await Promise.all([
          appFetch('/api/health'),
          appFetch('/api/models'),
        ])

        const healthPayload = healthResponse.ok
          ? (await healthResponse.json()) as HealthResponse
          : null

        if (!isCancelled) {
          setChatStorageAvailable(Boolean(healthPayload?.chat_storage_available))
          setBackendVersion(healthPayload?.version ?? null)
        }

        if (!modelsResponse.ok) {
          throw new Error('Nie udało się pobrać listy modeli.')
        }

        const modelsPayload = await modelsResponse.json() as ModelsResponse
        if (!isCancelled) {
          setAvailableModels(modelsPayload.models)
          setSelectedModel(modelsPayload.default_model)
        }

        const accountPayload = await refreshAccount(activeSession)

        if (healthPayload?.chat_storage_available) {
          const threadsResponse = await apiFetch('/api/chats', activeSession)
          if (!threadsResponse.ok) {
            throw new Error(await readErrorResponse(threadsResponse))
          }

          const threadsPayload = await threadsResponse.json() as ThreadsResponse
          if (!isCancelled) {
            setActiveThreads(threadsPayload.active)
            setArchivedThreads(threadsPayload.archived)
            const firstThread = threadsPayload.active[0] ?? threadsPayload.archived[0]
            const workspace = activeWorkspaceRef.current
            const canAutoSelectThread = (
              !workspace.selectedChatId
              && !workspace.hasMessages
              && !workspace.draft.trim()
              && !workspace.isSending
            )
            if (firstThread && canAutoSelectThread) {
              setSelectedChatId(firstThread.id)
            }
          }
        }

        if (!isCancelled) {
          setAccount(accountPayload)
        }

        if (!isCancelled) {
          await reconcileCheckoutReturn(activeSession)
        }

        if (accountPayload.is_admin) {
          setIsAdminUsersLoading(true)
          try {
            await refreshAdminUsers(activeSession)
          } catch (adminUsersError) {
            if (!isCancelled) {
              setAdminUsers([])
              setAdminUsersError(
                adminUsersError instanceof Error
                  ? adminUsersError.message
                  : 'Nie udało się wczytać użytkowników.',
              )
            }
          } finally {
            if (!isCancelled) {
              setIsAdminUsersLoading(false)
            }
          }
        } else if (!isCancelled) {
          setAdminUsers([])
          setAdminUsersError('')
        }
      } catch (bootstrapError) {
        if (!isCancelled) {
          setError(bootstrapError instanceof Error ? bootstrapError.message : 'Nie udało się uruchomić aplikacji.')
        }
      } finally {
        if (!isCancelled) {
          setIsBootstrapping(false)
        }
      }
    }

    void bootstrap()

    return () => {
      isCancelled = true
    }
  }, [session])

  useEffect(() => {
    if (!session || !selectedChatId) {
      return
    }

    if (!chatStorageAvailable && !isLocalThreadId(selectedChatId)) {
      setSelectedChatId(null)
      setMessages([])
      setIsThreadLoading(false)
      return
    }

    if (isLocalThreadId(selectedChatId)) {
      setMessages(localThreadMessages[selectedChatId] ?? [])
      setIsThreadLoading(false)
      setError('')
      return
    }

    let isCancelled = false
    setIsThreadLoading(true)
    setError('')

    void apiFetch(`/api/chats/${selectedChatId}`, session)
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(await readErrorResponse(response))
        }

        return response.json() as Promise<ThreadDetail>
      })
      .then((payload) => {
        if (!isCancelled) {
          setMessages(payload.messages)
        }
      })
      .catch((loadError) => {
        if (!isCancelled) {
          setMessages([])
          setError(loadError instanceof Error ? loadError.message : 'Nie udało się wczytać wątku.')
        }
      })
      .finally(() => {
        if (!isCancelled) {
          setIsThreadLoading(false)
        }
      })

    return () => {
      isCancelled = true
    }
  }, [chatStorageAvailable, localThreadMessages, selectedChatId, session])

  useEffect(() => {
    activeHintRequestControllerRef.current?.abort()
    activeHintRequestControllerRef.current = null

    const trimmedDraft = draft.trim()
    const answeredHints = buildIntentHintsPayload(intentHintAnswers)

    if (trimmedDraft.length < MIN_DRAFT_LENGTH_FOR_HINTS || isSending || answeredHints.length >= MAX_HINT_QUESTION_COUNT) {
      setPromptHints([])
      if (trimmedDraft.length < MIN_DRAFT_LENGTH_FOR_HINTS) {
        setIntentHintAnswers((currentAnswers) => (Object.keys(currentAnswers).length ? {} : currentAnswers))
      }
      setIsHintsLoading(false)
      setHintMode('fallback')
      return
    }

    const controller = new AbortController()
    activeHintRequestControllerRef.current = controller
    const timeoutId = window.setTimeout(() => {
      setIsHintsLoading(true)

      void fetchPromptHints({
        draftText: trimmedDraft,
        answeredHints,
        excludedQuestions: answeredHints.map((hint) => hint.question),
        maxHints: Math.min(ACTIVE_HINT_COUNT, Math.max(MAX_HINT_QUESTION_COUNT - answeredHints.length, 0)) || 1,
        signal: controller.signal,
      })
        .then((payload) => {
          if (controller.signal.aborted) {
            return
          }
          setPromptHints(payload.hints)
          setHintMode(payload.mode)
        })
        .catch(() => {
          if (controller.signal.aborted) {
            return
          }
          setPromptHints([])
          setHintMode('fallback')
        })
        .finally(() => {
          if (!controller.signal.aborted) {
            setIsHintsLoading(false)
          }
        })
    }, HINT_DEBOUNCE_MS)

    return () => {
      controller.abort()
      if (activeHintRequestControllerRef.current === controller) {
        activeHintRequestControllerRef.current = null
      }
      window.clearTimeout(timeoutId)
    }
  }, [draft, isSending, intentHintAnswers])

  function upsertActiveThread(thread: ThreadSummary) {
    setActiveThreads((currentThreads) => [
      thread,
      ...currentThreads.filter((currentThread) => currentThread.id !== thread.id),
    ])
    setArchivedThreads((currentThreads) =>
      currentThreads.filter((currentThread) => currentThread.id !== thread.id),
    )
  }

  function updateThreadSummary(threadId: string, updater: (thread: ThreadSummary) => ThreadSummary) {
    setActiveThreads((currentThreads) =>
      currentThreads.map((thread) => (thread.id === threadId ? updater(thread) : thread)),
    )
    setArchivedThreads((currentThreads) =>
      currentThreads.map((thread) => (thread.id === threadId ? updater(thread) : thread)),
    )
  }

  function replaceThreadId(previousId: string, nextThread: ThreadSummary) {
    setActiveThreads((currentThreads) => [
      nextThread,
      ...currentThreads.filter((thread) => thread.id !== previousId && thread.id !== nextThread.id),
    ])
    setArchivedThreads((currentThreads) =>
      currentThreads.filter((thread) => thread.id !== previousId && thread.id !== nextThread.id),
    )
  }

  async function refreshThreads(activeSession: Session, preferredChatId?: string | null) {
    if (!chatStorageAvailable) {
      setActiveThreads([])
      setArchivedThreads([])
      if (preferredChatId === null) {
        setSelectedChatId(null)
        setMessages([])
      }
      return
    }

    const response = await apiFetch('/api/chats', activeSession)
    if (!response.ok) {
      throw new Error(await readErrorResponse(response))
    }

    const payload = await response.json() as ThreadsResponse
    setActiveThreads(payload.active)
    setArchivedThreads(payload.archived)

    if (preferredChatId === null) {
      setSelectedChatId(null)
      setMessages([])
      return
    }

    const existingSelection = preferredChatId ?? selectedChatId
    const allThreads = [...payload.active, ...payload.archived]
    const nextSelection = existingSelection
      ? allThreads.find((thread) => thread.id === existingSelection)?.id
      : null

    if (nextSelection) {
      setSelectedChatId(nextSelection)
      return
    }

    const firstThread = payload.active[0] ?? payload.archived[0]
    setSelectedChatId(firstThread?.id ?? null)
    if (!firstThread) {
      setMessages([])
    }
  }

  async function handleAssistantFeedback(messageId: string, payload: ChatMessageFeedbackRequest) {
    if (!session || !selectedChatId || isLocalThreadId(selectedChatId)) {
      return
    }

    setFeedbackSavingMessageId(messageId)
    setError('')

    try {
      const response = await apiFetch(`/api/chats/${selectedChatId}/messages/${messageId}/feedback`, session, {
        method: 'POST',
        body: JSON.stringify({
          rating: payload.rating,
          comment: payload.comment?.trim() || null,
        }),
      })

      if (!response.ok) {
        throw new Error(await readErrorResponse(response))
      }

      const savedMessage = await response.json() as Message
      setMessages((currentMessages) =>
        currentMessages.map((message) => (message.id === messageId ? { ...message, ...savedMessage } : message)),
      )
      setLocalThreadMessages((currentThreads) => {
        if (!selectedChatId || !currentThreads[selectedChatId]) {
          return currentThreads
        }

        return {
          ...currentThreads,
          [selectedChatId]: currentThreads[selectedChatId].map((message) =>
            message.id === messageId ? { ...message, ...savedMessage } : message,
          ),
        }
      })
    } catch (feedbackError) {
      setError(feedbackError instanceof Error ? feedbackError.message : 'Nie udało się zapisać oceny odpowiedzi.')
    } finally {
      setFeedbackSavingMessageId(null)
    }
  }

  async function handleAuthSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!supabaseBrowserClient) {
      return
    }

    setAuthSubmitting(true)
    setAuthError('')
    setAuthInfo('')

    try {
      if (authMode === 'signup') {
        const { error: signUpError } = await supabaseBrowserClient.auth.signUp({
          email: authEmail.trim(),
          password: authPassword,
          options: {
            data: {
              full_name: authFullName.trim(),
            },
          },
        })

        if (signUpError) {
          throw signUpError
        }

        setAuthInfo('Konto zostało utworzone. Jeśli masz włączone potwierdzenie mailowe, dokończ je w skrzynce.')
      } else {
        const { error: signInError } = await supabaseBrowserClient.auth.signInWithPassword({
          email: authEmail.trim(),
          password: authPassword,
        })

        if (signInError) {
          throw signInError
        }
      }
    } catch (submitError) {
      setAuthError(submitError instanceof Error ? submitError.message : 'Nie udało się uwierzytelnić.')
    } finally {
      setAuthSubmitting(false)
    }
  }

  async function handleSignOut() {
    if (!supabaseBrowserClient) {
      return
    }

    setError('')
    await supabaseBrowserClient.auth.signOut()
  }

  async function handleArchive(chatId: string, archived: boolean) {
    if (!session) {
      return
    }

    if (isLocalThreadId(chatId)) {
      const source = activeThreads.find((thread) => thread.id === chatId) ?? archivedThreads.find((thread) => thread.id === chatId)
      if (!source) {
        return
      }

      const nextThread = { ...source, archived }
      if (archived) {
        setActiveThreads((currentThreads) => currentThreads.filter((thread) => thread.id !== chatId))
        setArchivedThreads((currentThreads) => [nextThread, ...currentThreads.filter((thread) => thread.id !== chatId)])
        if (selectedChatId === chatId) {
          setSelectedChatId(null)
          setMessages([])
        }
      } else {
        setArchivedThreads((currentThreads) => currentThreads.filter((thread) => thread.id !== chatId))
        setActiveThreads((currentThreads) => [nextThread, ...currentThreads.filter((thread) => thread.id !== chatId)])
      }

      return
    }

    setIsSidebarBusy(true)
    setError('')

    try {
      const response = await apiFetch(`/api/chats/${chatId}`, session, {
        method: 'PATCH',
        body: JSON.stringify({ archived }),
      })

      if (!response.ok) {
        throw new Error(await readErrorResponse(response))
      }

      const nextSelectedId = archived && selectedChatId === chatId ? null : chatId
      await refreshThreads(session, nextSelectedId)
    } catch (archiveError) {
      setError(archiveError instanceof Error ? archiveError.message : 'Nie udało się zaktualizować wątku.')
    } finally {
      setIsSidebarBusy(false)
    }
  }

  function handleStartNewChat() {
    activeHintRequestControllerRef.current?.abort()
    activeHintRequestControllerRef.current = null
    setSelectedChatId(null)
    setMessages([])
    setDraft('')
    setError('')
    setLastRedactions([])
    setPromptHints([])
    setIntentHintAnswers({})
    setHintMode('fallback')
  }

  function handleHintAnswer(hint: PromptHint, option: PromptHintOption, index: number) {
    const trimmedDraft = draft.trim()
    if (!trimmedDraft) {
      return
    }

    const nextAnswer: IntentHintAnswer = {
      question: hint.question,
      option_id: option.id,
      option_label: option.label,
    }
    const answeredHints = [
      ...buildIntentHintsPayload(intentHintAnswers),
      nextAnswer,
    ]

    setIntentHintAnswers((currentAnswers) => ({
      ...currentAnswers,
      [hint.id]: nextAnswer,
    }))

    const nextPromptHints = promptHints.filter((_, promptHintIndex) => promptHintIndex !== index)

    if (answeredHints.length >= MAX_HINT_QUESTION_COUNT) {
      setPromptHints(nextPromptHints)
      return
    }

    // Keep a single request path: updating the answer restarts the debounced
    // effect above and cancels the prior browser request.
    setPromptHints(nextPromptHints)
    setIsHintsLoading(true)
  }

  async function handleAdminGrantCredits(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!session) {
      return
    }

    const parsedAmount = Number.parseInt(adminGrantDraft.credit_amount, 10)
    if (!Number.isFinite(parsedAmount) || parsedAmount <= 0) {
      setError('Podaj dodatnia liczbe kredytow.')
      return
    }

    setIsAdminGrantSubmitting(true)
    setError('')
    setAdminGrantInfo('')

    try {
      const response = await apiFetch('/api/admin/credits/grant', session, {
        method: 'POST',
        body: JSON.stringify({
          user_email: adminGrantDraft.user_email.trim(),
          credit_amount: parsedAmount,
          reason: adminGrantDraft.reason.trim() || null,
        }),
      })

      if (!response.ok) {
        throw new Error(await readErrorResponse(response))
      }

      const payload = await response.json() as AdminGrantCreditsResponse
      setAdminGrantInfo(
        `Przyznano ${parsedAmount} kredytów dla ${payload.email ?? payload.user_id}. Nowe saldo: ${payload.credit_balance}.`,
      )
      setAdminGrantDraft({ user_email: '', credit_amount: '1', reason: '' })
      await refreshAccount(session)
      try {
        await refreshAdminUsers(session)
      } catch (adminUsersRefreshError) {
        setAdminUsers([])
        setAdminUsersError(
          adminUsersRefreshError instanceof Error
            ? adminUsersRefreshError.message
            : 'Nie udało się odświeżyć listy użytkowników.',
        )
      }
    } catch (grantError) {
      setError(grantError instanceof Error ? grantError.message : 'Nie udało się przyznać kredytów.')
    } finally {
      setIsAdminGrantSubmitting(false)
    }
  }

  async function handleQuickGrant(user: AdminUserSummary, creditAmount: number) {
    if (!session) {
      return
    }

    setIsAdminGrantSubmitting(true)
    setError('')
    setAdminGrantInfo('')

    try {
      const response = await apiFetch('/api/admin/credits/grant', session, {
        method: 'POST',
        body: JSON.stringify({
          user_email: user.email,
          credit_amount: creditAmount,
          reason: `Szybki grant z panelu admina (+${creditAmount})`,
        }),
      })

      if (!response.ok) {
        throw new Error(await readErrorResponse(response))
      }

      const payload = await response.json() as AdminGrantCreditsResponse
      setAdminGrantInfo(
        `Przyznano ${creditAmount} kredytów dla ${payload.email ?? payload.user_id}. Nowe saldo: ${payload.credit_balance}.`,
      )
      await refreshAccount(session)
      try {
        await refreshAdminUsers(session)
      } catch (adminUsersRefreshError) {
        setAdminUsers([])
        setAdminUsersError(
          adminUsersRefreshError instanceof Error
            ? adminUsersRefreshError.message
            : 'Nie udało się odświeżyć listy użytkowników.',
        )
      }
    } catch (grantError) {
      setError(grantError instanceof Error ? grantError.message : 'Nie udało się przyznać kredytów.')
    } finally {
      setIsAdminGrantSubmitting(false)
    }
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!session) {
      return
    }

    const trimmedDraft = draft.trim()
    if (!trimmedDraft || isSending) {
      return
    }

    activeHintRequestControllerRef.current?.abort()
    activeHintRequestControllerRef.current = null

    const nextMessages = [
      ...messages,
      { id: crypto.randomUUID(), role: 'user' as const, content: trimmedDraft },
    ]
    const intentHintsPayload = buildIntentHintsPayload(intentHintAnswers)

    let pendingChatId = selectedChatId
    if (!pendingChatId) {
      const optimisticThread = createOptimisticThread(trimmedDraft)
      pendingChatId = optimisticThread.id
      setSelectedChatId(optimisticThread.id)
      upsertActiveThread(optimisticThread)
      setLocalThreadMessages((currentMessages) => ({
        ...currentMessages,
        [optimisticThread.id]: nextMessages,
      }))
    } else {
      updateThreadSummary(pendingChatId, (thread) => ({
        ...thread,
        title: thread.title === 'Nowy wątek' ? buildThreadTitle(trimmedDraft) : thread.title,
        updated_at: new Date().toISOString(),
        last_message_preview: buildThreadPreview(trimmedDraft),
      }))

      if (isLocalThreadId(pendingChatId)) {
        setLocalThreadMessages((currentMessages) => ({
          ...currentMessages,
          [pendingChatId as string]: nextMessages,
        }))
      }
    }

    setMessages(nextMessages)
    setDraft('')
    setPromptHints([])
    setIntentHintAnswers({})
    setHintMode('fallback')
    setError('')
    setIsSending(true)

    try {
      const response = await apiFetch('/api/chat', session, {
        method: 'POST',
        body: JSON.stringify({
          messages: nextMessages.map(({ role, content }) => ({ role, content })),
          model: selectedModel,
          chat_id: isLocalThreadId(pendingChatId) ? null : pendingChatId,
          intent_hints: intentHintsPayload,
          retrieval_preferences: buildRetrievalPreferences(retrievalScope),
        }),
      })

      if (!response.ok) {
        throw new Error(await readErrorResponse(response))
      }

      const data = await response.json() as ChatResponse
      const assistantMessage = {
        id: data.assistant_message_id ?? crypto.randomUUID(),
        role: 'assistant' as const,
        content: data.reply,
        structured_reply: data.structured_reply ?? undefined,
        feedback_rating: null,
        feedback_comment: null,
        feedback_created_at: null,
      }
      const resolvedChatId = data.chat_id ?? pendingChatId
      const now = new Date().toISOString()

      setLastRedactions(data.redactions)

      if (pendingChatId && isLocalThreadId(pendingChatId)) {
        const nextLocalMessages = [...nextMessages, assistantMessage]
        if (data.chat_id && data.chat_id !== pendingChatId) {
          const persistedChatId = data.chat_id
          replaceThreadId(pendingChatId, {
            id: persistedChatId,
            title: buildThreadTitle(trimmedDraft),
            archived: false,
            created_at: now,
            updated_at: now,
            last_message_preview: buildThreadPreview(data.reply),
          })
          setLocalThreadMessages((currentMessages) => {
            const { [pendingChatId]: _removedThread, ...otherThreads } = currentMessages
            return {
              ...otherThreads,
              [persistedChatId]: nextLocalMessages,
            }
          })
        } else {
          updateThreadSummary(pendingChatId, (thread) => ({
            ...thread,
            updated_at: now,
            last_message_preview: buildThreadPreview(data.reply),
          }))
          setLocalThreadMessages((currentMessages) => ({
            ...currentMessages,
            [pendingChatId]: nextLocalMessages,
          }))
        }
      } else if (resolvedChatId) {
        updateThreadSummary(resolvedChatId, (thread) => ({
          ...thread,
          updated_at: now,
          last_message_preview: buildThreadPreview(data.reply),
        }))
      }

      if (resolvedChatId) {
        setSelectedChatId(resolvedChatId)
      }

      setMessages((currentMessages) => [
        ...currentMessages,
        assistantMessage,
      ])

      await refreshAccount(session)

      if (data.chat_id) {
        try {
          await refreshThreads(session, data.chat_id)
        } catch {
          // Keep optimistic thread list when storage is unavailable.
        }
      }
    } catch (submissionError) {
      setError(
        submissionError instanceof Error
          ? submissionError.message
          : 'Nie udało się wysłać wiadomości.',
      )
      if (session) {
        try {
          await refreshAccount(session)
        } catch {
          // Ignore secondary account refresh errors.
        }
      }
    } finally {
      setIsSending(false)
    }
  }

  const selectedThread = [...activeThreads, ...archivedThreads].find(
    (thread) => thread.id === selectedChatId,
  )
  const hasMessages = messages.length > 0
  const answeredHintCount = Object.keys(intentHintAnswers).length
  const totalHintCount = MAX_HINT_QUESTION_COUNT
  const remainingHintCount = Math.max(totalHintCount - answeredHintCount, 0)
  const latestAssistantMessageId = [...messages].reverse().find((message) => message.role === 'assistant')?.id ?? null
  const userDisplayName =
    account?.profile.full_name
    || session?.user.user_metadata?.full_name
    || session?.user.email
    || 'Konto'
  const isAdminUser = Boolean(account?.is_admin)

  if (!isSupabaseConfigured) {
    return (
      <main className="setup-shell">
        <section className="setup-card">
          <img className="setup-logo" src={logoUrl} alt="Alitigator" />
          <p className="eyebrow">Konfiguracja</p>
          <h1>Supabase nie jest jeszcze podpięty</h1>
          <p>
            Uzupełnij `VITE_SUPABASE_URL` i `VITE_SUPABASE_PUBLISHABLE_KEY`, żeby uruchomić logowanie,
            profile i rozliczanie kredytów.
          </p>
        </section>
      </main>
    )
  }

  if (authLoading) {
    return (
      <main className="setup-shell">
        <section className="setup-card">
          <img className="setup-logo" src={logoUrl} alt="Alitigator" />
          <p className="eyebrow">Sesja</p>
          <h1>Przygotowuję konto…</h1>
        </section>
      </main>
    )
  }

  if (!session) {
    return (
      <main className="auth-shell">
        <section className="auth-hero">
          <div className="brand-lockup brand-lockup-hero">
            <span className="brand-mark">
              <img className="brand-logo" src={logoUrl} alt="Alitigator" />
            </span>
            <div className="brand-meta">
              <strong>Alitigator</strong>
              <p>Research podatkowy z kontem użytkownika, historią i pakietami kredytów.</p>
            </div>
          </div>
          <div className="hero-copy">
            <p className="eyebrow">Konta i billing</p>
            <h1>Zaloguj się, żeby pracować na własnym saldzie kredytów</h1>
            <p>
              W tej wersji każdy użytkownik ma własny profil, historię wątków i saldo kredytów, które
              backend rozlicza po jednym kredycie za każde zapytanie.
            </p>
          </div>
        </section>

        <section className="auth-card">
          <div className="auth-tabs" role="tablist" aria-label="Tryb logowania">
            <button
              type="button"
              className={authMode === 'signin' ? 'auth-tab auth-tab-active' : 'auth-tab'}
              onClick={() => setAuthMode('signin')}
            >
              Logowanie
            </button>
            <button
              type="button"
              className={authMode === 'signup' ? 'auth-tab auth-tab-active' : 'auth-tab'}
              onClick={() => setAuthMode('signup')}
            >
              Rejestracja
            </button>
          </div>

          <form className="auth-form" onSubmit={handleAuthSubmit}>
            {authMode === 'signup' ? (
              <label className="auth-field">
                <span>Imię i nazwisko</span>
                <input
                  value={authFullName}
                  onChange={(event) => setAuthFullName(event.target.value)}
                  placeholder="np. Anna Kowalska"
                  autoComplete="name"
                />
              </label>
            ) : null}

            <label className="auth-field">
              <span>E-mail</span>
              <input
                type="email"
                value={authEmail}
                onChange={(event) => setAuthEmail(event.target.value)}
                placeholder="anna@kancelaria.pl"
                autoComplete="email"
                required
              />
            </label>

            <label className="auth-field">
              <span>Hasło</span>
              <input
                type="password"
                value={authPassword}
                onChange={(event) => setAuthPassword(event.target.value)}
                placeholder="Minimum 6 znaków"
                autoComplete={authMode === 'signup' ? 'new-password' : 'current-password'}
                required
              />
            </label>

            {authError ? <p className="error-text">{authError}</p> : null}
            {authInfo ? <p className="helper-text helper-text-positive">{authInfo}</p> : null}

            <button type="submit" className="auth-submit" disabled={authSubmitting}>
              {authSubmitting
                ? 'Przetwarzam…'
                : authMode === 'signup'
                  ? 'Załóż konto'
                  : 'Zaloguj się'}
            </button>
          </form>
        </section>
      </main>
    )
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="sidebar-topbar">
          <div className="brand-lockup">
            <span className="brand-mark">
              <img className="brand-logo" src={logoUrl} alt="Alitigator" />
            </span>
            <div className="brand-meta">
              <strong>Alitigator</strong>
              <p>{userDisplayName}</p>
            </div>
          </div>
          <button type="button" className="sidebar-primary-action" onClick={handleStartNewChat}>
            <SidebarActionIcon kind="new" />
            <span>Nowy wątek</span>
          </button>
        </div>

        <div className="thread-column">
          <div className="thread-section-header">
            <span>Wątki</span>
            <span className="thread-count">{activeThreads.length}</span>
          </div>

          <div className="thread-list" role="list" aria-label="Aktywne wątki">
            {activeThreads.map((thread) => (
              <button
                key={thread.id}
                type="button"
                className={`thread-item ${selectedChatId === thread.id ? 'thread-item-active' : ''}`}
                onClick={() => setSelectedChatId(thread.id)}
              >
                <div className="thread-item-copy">
                  <strong>{thread.title}</strong>
                  <p>{thread.last_message_preview || 'Pusty wątek'}</p>
                </div>
                <span
                  className="thread-item-action"
                  onClick={(event) => {
                    event.stopPropagation()
                    void handleArchive(thread.id, true)
                  }}
                >
                  <SidebarActionIcon kind="archive" />
                </span>
                <div className="thread-item-meta">
                  <span>{formatThreadTimestamp(thread.updated_at)}</span>
                </div>
              </button>
            ))}

            {!isBootstrapping && activeThreads.length === 0 ? (
              <div className="thread-empty-state">
                <p>Tu będą Twoje bieżące rozmowy.</p>
              </div>
            ) : null}
          </div>

          <div className="thread-archive-shell">
            <button
              type="button"
              className="thread-archive-toggle"
              onClick={() => setShowArchived((current) => !current)}
            >
              <span>Archiwum</span>
              <span className="thread-count">{archivedThreads.length}</span>
            </button>

            {showArchived ? (
              <div className="thread-list" role="list" aria-label="Archiwalne wątki">
                {archivedThreads.map((thread) => (
                  <button
                    key={thread.id}
                    type="button"
                    className={`thread-item ${selectedChatId === thread.id ? 'thread-item-active' : ''}`}
                    onClick={() => setSelectedChatId(thread.id)}
                  >
                    <div className="thread-item-copy">
                      <strong>{thread.title}</strong>
                      <p>{thread.last_message_preview || 'Brak podglądu'}</p>
                    </div>
                    <span
                      className="thread-item-action"
                      onClick={(event) => {
                        event.stopPropagation()
                        void handleArchive(thread.id, false)
                      }}
                    >
                      <SidebarActionIcon kind="restore" />
                    </span>
                    <div className="thread-item-meta">
                      <span>{formatThreadTimestamp(thread.updated_at)}</span>
                    </div>
                  </button>
                ))}
              </div>
            ) : null}
          </div>
        </div>

        <div className="sidebar-account">
          <div className="status-chip status-chip-wide">
            <span className="status-icon"><StatusGlyph kind="account" /></span>
            <span>
              {isAdminUser ? (
                <button
                  type="button"
                  className="admin-email-trigger"
                  onClick={() => setIsAdminPanelOpen((current) => !current)}
                  aria-expanded={isAdminPanelOpen}
                >
                  {account?.email ?? session.user.email}
                </button>
              ) : (
                <strong>{account?.email ?? session.user.email}</strong>
              )}
              <small>Konto aktywne</small>
            </span>
          </div>

          <button type="button" className="ghost-button" onClick={() => void handleSignOut()}>
            <SidebarActionIcon kind="logout" />
            <span>Wyloguj</span>
          </button>
        </div>
      </aside>

      <section className="chat-panel">
        <header className="chat-header">
          <div className="chat-heading">
            <div className="chat-heading-copy">
              <p className="eyebrow">Alitigator</p>
              <h2>{selectedThread?.title ?? 'Nowy wątek'}</h2>
            </div>
            {selectedThread ? <p className="chat-heading-meta">Ostatnia aktywność {formatThreadTimestamp(selectedThread.updated_at)}</p> : null}
          </div>

          <div className="status-tray" aria-label="Status aplikacji">
            <div className="status-chip">
              <span className="status-icon"><StatusGlyph kind="credits" /></span>
              <span>
                <strong>{formatTokenCount(account?.credit_balance ?? 0)}</strong>
                <small>Kredyty</small>
              </span>
            </div>
          </div>
        </header>

        <div className="message-list">
          {isThreadLoading ? <div className="chat-empty-state"><p>Wczytuję rozmowę…</p></div> : null}
          {!isThreadLoading && !hasMessages ? (
            <div className="chat-empty-state">
              <p>Zacznij od pytania albo otwórz wcześniejszy wątek z lewej strony.</p>
            </div>
          ) : null}

          {!isThreadLoading
            ? messages.map((message) => (
                <article key={message.id} className={`message message-${message.role}`}>
                  <p className="message-role">{message.role === 'assistant' ? 'aLitigator' : 'Ty'}</p>
                  {message.role === 'assistant' ? (
                    <>
                      <AssistantMessageBody
                        content={message.content}
                        structuredReply={message.structured_reply}
                      />
                      {message.feedback_rating ? (
                        <div className="message-feedback-card">
                          <p className="message-feedback-label">
                            Ocena jakości: <strong>{message.feedback_rating}/5</strong>
                          </p>
                          {message.feedback_comment ? (
                            <p className="message-feedback-note">{message.feedback_comment}</p>
                          ) : null}
                        </div>
                      ) : message.id === latestAssistantMessageId ? (
                        <div className="message-feedback-card">
                          <div className="message-feedback-header">
                            <p className="message-feedback-label">Oceń jakość tej odpowiedzi</p>
                            <small>Pomoże nam poprawiać kolejne odpowiedzi i retrieval.</small>
                          </div>
                          <div className="message-feedback-actions" role="group" aria-label="Ocena jakości odpowiedzi">
                            {[1, 2, 3, 4, 5].map((rating) => (
                              <button
                                key={rating}
                                type="button"
                                className="message-feedback-score"
                                disabled={feedbackSavingMessageId === message.id}
                                onClick={() => void handleAssistantFeedback(message.id, {
                                  rating,
                                  comment: feedbackDrafts[message.id] ?? '',
                                })}
                              >
                                {rating}
                              </button>
                            ))}
                          </div>
                          <label className="message-feedback-comment">
                            <span>Komentarz opcjonalny</span>
                            <textarea
                              value={feedbackDrafts[message.id] ?? ''}
                              onChange={(event) => setFeedbackDrafts((currentDrafts) => ({
                                ...currentDrafts,
                                [message.id]: event.target.value,
                              }))}
                              rows={3}
                              maxLength={1200}
                              placeholder="Co było trafne, czego zabrakło, co było za mało precyzyjne?"
                            />
                          </label>
                        </div>
                      ) : null}
                    </>
                  ) : (
                    <p className="message-content">{message.content}</p>
                  )}
                </article>
              ))
            : null}
          {isSending ? (
            <article className="message message-assistant message-loading">
              <p className="message-role">aLitigator</p>
              <p className="message-content">Analizuję pytanie...</p>
            </article>
          ) : null}
        </div>

        <footer className="composer-shell">
          {lastRedactions.length > 0 ? (
            <p className="helper-text">
              Ostatnie zapytanie zostało przefiltrowane pod kątem: {lastRedactions.join(', ')}.
            </p>
          ) : null}

          {error ? <p className="error-text">{error}</p> : null}

          <form className="composer" onSubmit={handleSubmit}>
            <div className="composer-toolbar composer-toolbar-compact">
              <div className="model-select-shell">
                <label className="model-select-label" htmlFor="model-select">
                  Model
                </label>
                <div className="model-select-wrap">
                  <select
                    id="model-select"
                    name="model-select"
                    value={selectedModel}
                    onChange={(event) => setSelectedModel(event.target.value)}
                    disabled={isSending}
                  >
                    {availableModels.map((model) => (
                      <option key={model} value={model}>
                        {modelLabels[model] ?? model}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </div>

            <label className="sr-only" htmlFor="chat-input">
              Treść pytania
            </label>
            <textarea
              id="chat-input"
              name="chat-input"
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              placeholder="Opisz stan faktyczny, pytanie podatkowe albo szkic pisma do rozwinięcia..."
              rows={6}
              maxLength={12000}
            />

            {draft.trim().length >= MIN_DRAFT_LENGTH_FOR_HINTS ? (
              <section className="prompt-hints-panel" aria-live="polite">
                <div className="prompt-hints-header">
                  <div>
                    <p className="prompt-hints-eyebrow">Tryb podpowiedzi</p>
                    <strong>Pomóż nam lepiej odczytać intencję pytania</strong>
                    <p className="prompt-hints-progress">
                      Zebrane: {answeredHintCount} z {totalHintCount}. Pozostało: {remainingHintCount}.
                    </p>
                  </div>
                  <span className={`prompt-hints-badge prompt-hints-badge-${hintMode}`}>
                    {isHintsLoading ? 'Analiza…' : hintMode === 'live' ? 'Haiku' : 'Fallback'}
                  </span>
                </div>

                {promptHints.length > 0 ? (
                  <div className="prompt-hints-list">
                    {promptHints.map((hint, index) => (
                      <article key={hint.id} className="prompt-hint-card">
                        <p className="prompt-hint-question">{hint.question}</p>
                        <div className="prompt-hint-options" role="group" aria-label={hint.question}>
                          {hint.options.map((option) => (
                            <button
                              key={option.id}
                              type="button"
                              className={`prompt-hint-option ${intentHintAnswers[hint.id]?.option_id === option.id ? 'prompt-hint-option-active' : ''}`}
                              onClick={() => handleHintAnswer(hint, option, index)}
                            >
                              {option.label}
                            </button>
                          ))}
                        </div>
                      </article>
                    ))}
                  </div>
                ) : (
                  <p className="helper-text small-text">
                    {isHintsLoading
                      ? 'Przygotowuję krótkie pytania doprecyzowujące…'
                      : 'Gdy treść będzie bardziej konkretna, pokażemy tu pytania pomocnicze.'}
                  </p>
                )}
              </section>
            ) : null}

            <div className="composer-actions">
              <button
                type="submit"
                className="submit-button submit-button-icon"
                disabled={isSending || isSidebarBusy}
                aria-label={isSending ? 'Analizuję' : 'Wyślij'}
              >
                <SendIcon />
              </button>
            </div>
          </form>
          <p className="helper-text small-text app-version-footer">
            Wersja {APP_VERSION} · API {backendVersion ?? 'niedostępne'}
          </p>
        </footer>
      </section>

      {isAdminUser && isAdminPanelOpen ? (
        <div className="admin-panel-overlay" role="dialog" aria-modal="true" aria-label="Panel administratora">
          <section className="admin-panel-modal">
            <div className="admin-panel-header">
              <div className="admin-panel-heading">
                <p className="eyebrow">Admin</p>
                <h2>Panel kredytów</h2>
                <p className="admin-panel-intro">
                  Podgląd salda użytkowników i ręczne przyznawanie kredytów bez checkoutu.
                </p>
              </div>
              <button
                type="button"
                className="ghost-button"
                onClick={() => setIsAdminPanelOpen(false)}
              >
                Zamknij
              </button>
            </div>

            {adminGrantInfo ? <p className="helper-text helper-text-positive">{adminGrantInfo}</p> : null}
            {adminUsersError ? <p className="helper-text helper-text-warning">{adminUsersError}</p> : null}

            <div className="admin-panel-grid">
              <div className="admin-panel-section">
                <div className="admin-panel-section-header">
                  <div>
                    <p className="eyebrow">Użytkownicy</p>
                    <h3>Zarejestrowane konta</h3>
                  </div>
                  <span className="admin-panel-badge">{adminUsers.length}</span>
                </div>

                <div className="admin-user-list">
                  {isAdminUsersLoading ? (
                    <div className="admin-empty-state">
                      <p className="helper-text small-text">Wczytuję użytkowników…</p>
                    </div>
                  ) : null}
                  {!isAdminUsersLoading && adminUsers.length === 0 ? (
                    <div className="admin-empty-state">
                      <p className="helper-text small-text">Brak użytkowników do wyświetlenia.</p>
                    </div>
                  ) : null}
                  {adminUsers.map((user) => (
                    <article key={user.user_id} className="admin-user-card">
                      <div className="admin-user-copy">
                        <div className="admin-user-heading">
                          <strong>{user.full_name || user.email || user.user_id}</strong>
                          {user.is_admin ? <span className="admin-user-role">Admin</span> : null}
                        </div>
                        <small>{user.email || 'Brak e-maila'}</small>
                        <small>{user.law_firm || 'Bez kancelarii'}</small>
                      </div>
                      <div className="admin-user-balance">
                        <strong>{formatTokenCount(user.credit_balance)}</strong>
                        <small>kredytów</small>
                      </div>
                      <div className="admin-user-actions">
                        <button
                          type="button"
                          className="secondary-button"
                          disabled={isAdminGrantSubmitting || !user.email}
                          onClick={() => void handleQuickGrant(user, 1)}
                        >
                          +1
                        </button>
                        <button
                          type="button"
                          className="secondary-button"
                          disabled={isAdminGrantSubmitting || !user.email}
                          onClick={() => void handleQuickGrant(user, 5)}
                        >
                          +5
                        </button>
                        <button
                          type="button"
                          className="secondary-button"
                          disabled={isAdminGrantSubmitting || !user.email}
                          onClick={() => void handleQuickGrant(user, 10)}
                        >
                          +10
                        </button>
                        <button
                          type="button"
                          className="ghost-button"
                          disabled={isAdminGrantSubmitting || !user.email}
                          onClick={() => setAdminGrantDraft((current) => ({
                            ...current,
                            user_email: user.email ?? current.user_email,
                          }))}
                        >
                          Własna kwota
                        </button>
                      </div>
                    </article>
                  ))}
                </div>
              </div>

              <div className="admin-panel-section admin-panel-section-form">
                <div className="admin-panel-section-header">
                  <div>
                    <p className="eyebrow">Akcja</p>
                    <h3>Przyznaj kredyty</h3>
                  </div>
                  <span className="admin-panel-badge admin-panel-badge-accent">Manual</span>
                </div>

                <form className="admin-grant-form" onSubmit={handleAdminGrantCredits}>
                  <label className="profile-field">
                    <span>E-mail użytkownika</span>
                    <input
                      type="email"
                      value={adminGrantDraft.user_email}
                      onChange={(event) => setAdminGrantDraft((current) => ({ ...current, user_email: event.target.value }))}
                      placeholder="anna@kancelaria.pl"
                      required
                    />
                  </label>
                  <label className="profile-field">
                    <span>Liczba kredytów</span>
                    <input
                      type="number"
                      min={1}
                      step={1}
                      value={adminGrantDraft.credit_amount}
                      onChange={(event) => setAdminGrantDraft((current) => ({ ...current, credit_amount: event.target.value }))}
                      required
                    />
                  </label>
                  <label className="profile-field">
                    <span>Powód</span>
                    <input
                      value={adminGrantDraft.reason}
                      onChange={(event) => setAdminGrantDraft((current) => ({ ...current, reason: event.target.value }))}
                      placeholder="np. grant testowy / reklamacja"
                    />
                  </label>
                  <button type="submit" className="secondary-button" disabled={isAdminGrantSubmitting}>
                    {isAdminGrantSubmitting ? 'Przyznaję…' : 'Przyznaj kredyty'}
                  </button>
                </form>
              </div>
            </div>
          </section>
        </div>
      ) : null}
    </main>
  )
}

export default App
