import { useEffect, useState } from 'react'
import type { FormEvent } from 'react'
import './App.css'
import { isSupabaseConfigured } from './lib/supabase'
import logoUrl from './assets/alitigator-logo.png'

type Role = 'user' | 'assistant'

type Message = {
  id: string
  role: Role
  content: string
}

type ChatResponse = {
  reply: string
  mode: 'demo' | 'live'
  model: string
  redactions: string[]
}

type ModelsResponse = {
  default_model: string
  models: string[]
}

const modelLabels: Record<string, string> = {
  'claude-opus-4-8': 'Claude Opus 4.8',
  'claude-sonnet-4-6': 'Claude Sonnet 4.6',
  'claude-haiku-4-5-20251001': 'Claude Haiku 4.5',
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

const initialMessages: Message[] = [
  {
    id: crypto.randomUUID(),
    role: 'assistant',
    content:
      'Jestem prototypem aLitigator. Na tym etapie pomagam prowadzić prostą rozmowę roboczą dla prawnika podatkowego. Docelowo odpowiem w oparciu o RAG z interpretacji, orzeczeń, ustaw i literatury.',
  },
]

function App() {
  const [messages, setMessages] = useState<Message[]>(initialMessages)
  const [draft, setDraft] = useState('')
  const [error, setError] = useState('')
  const [isSending, setIsSending] = useState(false)
  const [availableModels, setAvailableModels] = useState<string[]>(['claude-sonnet-4-6'])
  const [selectedModel, setSelectedModel] = useState('claude-sonnet-4-6')
  const [lastMode, setLastMode] = useState<'demo' | 'live'>('demo')
  const [lastModel, setLastModel] = useState('claude-sonnet-4-6')
  const [lastRedactions, setLastRedactions] = useState<string[]>([])

  const creditsLeft = 17

  useEffect(() => {
    let isCancelled = false

    void fetch(`${API_BASE_URL}/api/models`)
      .then(async (response) => {
        if (!response.ok) {
          throw new Error('Nie udało się pobrać listy modeli.')
        }

        return (await response.json()) as ModelsResponse
      })
      .then((payload) => {
        if (isCancelled) {
          return
        }

        setAvailableModels(payload.models)
        setSelectedModel(payload.default_model)
        setLastModel(payload.default_model)
      })
      .catch(() => {
        if (isCancelled) {
          return
        }

        setAvailableModels(['claude-sonnet-4-6'])
      })

    return () => {
      isCancelled = true
    }
  }, [])

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
        }),
      })

      if (!response.ok) {
        throw new Error(`Backend zwrócił ${response.status}`)
      }

      const data: ChatResponse = await response.json()
      setLastMode(data.mode)
      setLastModel(data.model)
      setLastRedactions(data.redactions)
      setMessages((currentMessages) => [
        ...currentMessages,
        { id: crypto.randomUUID(), role: 'assistant', content: data.reply },
      ])
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

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand-card">
          <img className="brand-logo" src={logoUrl} alt="aLitigator" />
          <div className="brand-copy">
            <p className="eyebrow">aLitigator MVP</p>
            <h1>Research podatkowy i szkice pism</h1>
            <p className="lede">
              Minimalny interfejs do rozmowy z asystentem. Priorytet na teraz to szybka praca,
              źródła i bezpieczny przepływ danych, nie marketingowy landing page.
            </p>
          </div>
        </div>

        <div className="meta-grid">
          <section className="meta-card compact-card">
            <p className="meta-label">Tryb</p>
            <strong>{lastMode === 'live' ? 'Claude live' : 'Demo fallback'}</strong>
            <p>{lastModel}</p>
          </section>

          <section className="meta-card compact-card">
            <p className="meta-label">Supabase</p>
            <strong>{isSupabaseConfigured ? 'Połączony' : 'Brak env'}</strong>
            <p>Schema `alitigator` gotowy pod profile i historię czatów.</p>
          </section>

          <section className="meta-card compact-card">
            <p className="meta-label">Kredyty</p>
            <strong>{creditsLeft} pytań</strong>
            <p>Statyczny placeholder przed prawdziwym billingiem.</p>
          </section>

          <section className="meta-card compact-card">
            <p className="meta-label">Bezpieczeństwo</p>
            <strong>Maskowanie aktywne</strong>
            <p>Email, PESEL, NIP i telefon są filtrowane po stronie backendu.</p>
          </section>
        </div>

        <section className="meta-card roadmap-card">
          <p className="meta-label">Dalej</p>
          <strong>RAG, fallback web, cytowania, sprawy</strong>
          <p>Najbliższa iteracja to porządne źródła i zapisywanie rozmów użytkownika.</p>
        </section>
      </aside>

      <section className="chat-panel">
        <header className="chat-header">
          <div>
            <p className="eyebrow">MVP chat</p>
            <h2>Okno robocze</h2>
          </div>
          <div className="header-badges">
            <span className="security-pill">Dane wrażliwe: maskowanie aktywne</span>
            <span className="security-pill muted-pill">RAG, potem fallback web</span>
          </div>
        </header>

        <div className="message-list">
          {messages.map((message) => (
            <article
              key={message.id}
              className={`message message-${message.role}`}
            >
              <p className="message-role">
                {message.role === 'assistant' ? 'aLitigator' : 'Ty'}
              </p>
              <p className="message-content">{message.content}</p>
            </article>
          ))}
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
            <p className="helper-text">
              W tym prototypie odpowiedzi mają strukturę: teza, analiza, źródła, ryzyka.
            </p>
          )}

          {error ? <p className="error-text">{error}</p> : null}

          <form className="composer" onSubmit={handleSubmit}>
            <div className="composer-toolbar">
              <label className="model-select-label" htmlFor="model-select">
                Model
              </label>
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
              <p className="helper-text small-text">
                Nie wklejaj pełnych danych klienta, dopóki nie dodamy pełnej polityki retencji i audytu.
              </p>
              <button type="submit" disabled={isSending || !draft.trim()}>
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
