import { useEffect, useState } from 'react'
import type { FormEvent } from 'react'
import './App.css'
import logoUrl from './assets/alitigator-logo.png'

type Role = 'user' | 'assistant'
type ChatMode = 'demo' | 'live'

type Message = {
  id: string
  role: Role
  content: string
  created_at?: string
}

type ChatResponse = {
  reply: string
  mode: ChatMode
  model: string
  redactions: string[]
  chat_id?: string
}

type ModelsResponse = {
  default_model: string
  models: string[]
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

const modelLabels: Record<string, string> = {
  'claude-opus-4-8': 'Claude Opus 4.8',
  'claude-sonnet-4-6': 'Claude Sonnet 4.6',
  'claude-haiku-4-5-20251001': 'Claude Haiku 4.5',
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'
const LOCAL_THREAD_PREFIX = 'local-thread-'
const APP_VERSION = '0.4.0'

const creditsLeft = 17

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

function StatusGlyph({ kind }: { kind: 'credits' | 'model' | 'shield' }) {
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

  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 3 5 6v6c0 4.4 2.8 8.5 7 9.8 4.2-1.3 7-5.4 7-9.8V6l-7-3Z" />
      <path d="m9.5 12 1.7 1.7 3.5-3.7" />
    </svg>
  )
}

function SidebarActionIcon({ kind }: { kind: 'new' | 'archive' | 'restore' }) {
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

  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 7h16" />
      <path d="M9 7V5h6v2" />
      <path d="M7 7l1 11h8l1-11" />
    </svg>
  )
}

function App() {
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
  const [availableModels, setAvailableModels] = useState<string[]>(['claude-sonnet-4-6'])
  const [selectedModel, setSelectedModel] = useState('claude-sonnet-4-6')
  const [lastMode, setLastMode] = useState<ChatMode>('demo')
  const [lastModel, setLastModel] = useState('claude-sonnet-4-6')
  const [lastRedactions, setLastRedactions] = useState<string[]>([])

  useEffect(() => {
    let isCancelled = false

    async function bootstrap() {
      try {
        const [modelsResponse, threadsResponse] = await Promise.all([
          fetch(`${API_BASE_URL}/api/models`),
          fetch(`${API_BASE_URL}/api/chats`),
        ])

        if (!modelsResponse.ok) {
          throw new Error('Nie udało się pobrać listy modeli.')
        }

        const modelsPayload = (await modelsResponse.json()) as ModelsResponse
        if (!isCancelled) {
          setAvailableModels(modelsPayload.models)
          setSelectedModel(modelsPayload.default_model)
          setLastModel(modelsPayload.default_model)
        }

        if (threadsResponse.ok) {
          const threadsPayload = (await threadsResponse.json()) as ThreadsResponse
          if (!isCancelled) {
            setActiveThreads(threadsPayload.active)
            setArchivedThreads(threadsPayload.archived)
            const firstThread = threadsPayload.active[0] ?? threadsPayload.archived[0]
            if (firstThread) {
              setSelectedChatId(firstThread.id)
            }
          }
        }
      } catch {
        if (!isCancelled) {
          setAvailableModels(['claude-sonnet-4-6'])
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
  }, [])

  useEffect(() => {
    if (!selectedChatId) {
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

    void fetch(`${API_BASE_URL}/api/chats/${selectedChatId}`)
      .then(async (response) => {
        if (!response.ok) {
          throw new Error('Nie udało się wczytać wątku.')
        }

        return (await response.json()) as ThreadDetail
      })
      .then((payload) => {
        if (isCancelled) {
          return
        }

        setMessages(payload.messages)
      })
      .catch((loadError) => {
        if (isCancelled) {
          return
        }

        setMessages([])
        setError(loadError instanceof Error ? loadError.message : 'Nie udało się wczytać wątku.')
      })
      .finally(() => {
        if (!isCancelled) {
          setIsThreadLoading(false)
        }
      })

    return () => {
      isCancelled = true
    }
  }, [localThreadMessages, selectedChatId])

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

  async function refreshThreads(preferredChatId?: string | null) {
    const response = await fetch(`${API_BASE_URL}/api/chats`)
    if (!response.ok) {
      throw new Error('Nie udało się odświeżyć listy wątków.')
    }

    const payload = (await response.json()) as ThreadsResponse
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

  async function handleArchive(chatId: string, archived: boolean) {
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
      const response = await fetch(`${API_BASE_URL}/api/chats/${chatId}`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ archived }),
      })

      if (!response.ok) {
        throw new Error('Nie udało się zaktualizować wątku.')
      }

      const nextSelectedId = archived && selectedChatId === chatId ? null : chatId
      await refreshThreads(nextSelectedId)
    } catch (archiveError) {
      setError(archiveError instanceof Error ? archiveError.message : 'Nie udało się zaktualizować wątku.')
    } finally {
      setIsSidebarBusy(false)
    }
  }

  function handleStartNewChat() {
    setSelectedChatId(null)
    setMessages([])
    setDraft('')
    setError('')
    setLastRedactions([])
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()

    const trimmedDraft = draft.trim()
    if (!trimmedDraft || isSending) {
      return
    }

    const nextMessages = [
      ...messages,
      { id: crypto.randomUUID(), role: 'user' as const, content: trimmedDraft },
    ]

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
    setError('')
    setIsSending(true)

    try {
      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          messages: nextMessages.map(({ role, content }) => ({ role, content })),
          model: selectedModel,
          chat_id: isLocalThreadId(pendingChatId) ? null : pendingChatId,
        }),
      })

      if (!response.ok) {
        throw new Error(`Backend zwrócił ${response.status}`)
      }

      const data: ChatResponse = await response.json()
      const assistantMessage = { id: crypto.randomUUID(), role: 'assistant' as const, content: data.reply }
      const resolvedChatId = data.chat_id ?? pendingChatId
      const now = new Date().toISOString()

      setLastMode(data.mode)
      setLastModel(data.model)
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

      if (data.chat_id) {
        try {
          await refreshThreads(data.chat_id)
        } catch {
          // Keep the optimistic sidebar state when history storage is unavailable.
        }
      }
    } catch (submissionError) {
      setError(
        submissionError instanceof Error
          ? submissionError.message
          : 'Nie udało się wysłać wiadomości.',
      )
    } finally {
      setIsSending(false)
    }
  }

  const selectedThread = [...activeThreads, ...archivedThreads].find(
    (thread) => thread.id === selectedChatId,
  )
  const hasMessages = messages.length > 0
  const modelStatusLabel = lastMode === 'live' ? 'Live' : 'Demo'
  const modelDisplayLabel = modelLabels[lastModel] ?? lastModel

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
                {archivedThreads.length === 0 ? (
                  <div className="thread-empty-state muted-empty-state">
                    <p>Jeszcze nic tu nie trafiło.</p>
                  </div>
                ) : null}
              </div>
            ) : null}
          </div>
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
                <strong>{creditsLeft}</strong>
                <small>Kredyty</small>
              </span>
            </div>
            <div className="status-chip">
              <span className="status-icon"><StatusGlyph kind="model" /></span>
              <span>
                <strong>{modelStatusLabel}</strong>
                <small>{modelDisplayLabel}</small>
              </span>
            </div>
            <div className="status-chip">
              <span className="status-icon"><StatusGlyph kind="shield" /></span>
              <span>
                <strong>Maskowanie</strong>
                <small>Aktywne</small>
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
                  <p className="message-content">{message.content}</p>
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
          ) : (
            <p className="helper-text">Odpowiedź wraca w układzie: teza, analiza, źródła, ryzyka.</p>
          )}

          {error ? <p className="error-text">{error}</p> : null}

          <form className="composer" onSubmit={handleSubmit}>
            <div className="composer-toolbar">
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
                  <span className="model-select-chevron" aria-hidden="true">
                    <svg viewBox="0 0 20 20">
                      <path d="m5 7 5 6 5-6" />
                    </svg>
                  </span>
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
              rows={5}
              maxLength={12000}
            />
            <div className="composer-actions">
              <div className="composer-footer-meta">
                <p className="helper-text small-text">
                  Jeśli możesz, opisuj stan faktyczny bez pełnych danych klienta.
                </p>
                <p className="version-badge">Wersja {APP_VERSION}</p>
              </div>
              <button type="submit" disabled={isSending || !draft.trim() || isSidebarBusy}>
                {isSending ? 'Wysyłanie...' : 'Wyślij pytanie'}
              </button>
            </div>
          </form>
        </footer>
      </section>
    </main>
  )
}

export default App
