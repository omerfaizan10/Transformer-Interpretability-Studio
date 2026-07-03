import { useEffect, useMemo, useState } from 'react'
import {
  Activity,
  Brain,
  Cpu,
  Eye,
  GitBranch,
  Layers3,
  Play,
  ShieldCheck,
  Sparkles,
  TerminalSquare,
} from 'lucide-react'

import {
  analyzeAttention,
  generateText,
  explainAttention,
  testClaude,
  getConfig,
  getHealth,
  getPositionalEncoding,
} from './api.js'

import Heatmap from './components/Heatmap.jsx'
import MetricCard from './components/MetricCard.jsx'
import TokenRibbon from './components/TokenRibbon.jsx'

const defaultAnalysis = {
  text: 'attention lets each token decide which earlier tokens matter',
  layer: 0,
  head: 0,
  causal_mask: true,
  analysis_mode: 'guided',
}

function TabButton({ active, children, onClick }) {
  return (
    <button
      onClick={onClick}
      className={`rounded-xl px-4 py-3 text-sm font-semibold transition ${
        active
          ? 'bg-white text-slate-950 shadow-lg shadow-slate-950/20'
          : 'border border-slate-800 bg-slate-950 text-slate-400 hover:border-slate-700 hover:text-slate-100'
      }`}
    >
      {children}
    </button>
  )
}

function SectionHeader({ eyebrow, title, children, icon: Icon }) {
  return (
    <div className="mb-7 grid gap-4 lg:grid-cols-[0.7fr_1fr] lg:items-end">
      <div>
        <div className="section-kicker flex items-center gap-2">
          {Icon ? <Icon size={15} /> : null}
          {eyebrow}
        </div>
        <h2 className="mt-2 text-3xl font-semibold tracking-tight text-white md:text-4xl">{title}</h2>
      </div>
      {children ? <p className="max-w-2xl text-sm leading-6 text-slate-400 lg:justify-self-end">{children}</p> : null}
    </div>
  )
}

function Connections({ links = [] }) {
  return (
    <div className="studio-card rounded-3xl p-5">
      <p className="section-kicker">Strongest token routes</p>
      <h3 className="mt-2 text-xl font-semibold text-white">Top Attention Connections</h3>

      <div className="mt-5 space-y-3">
        {links.map((link, index) => (
          <div key={`${link.query_index}-${link.key_index}-${index}`} className="rounded-2xl border border-slate-800 bg-slate-950 p-4">
            <div className="flex items-center justify-between gap-3">
              <div className="text-sm text-slate-400">
                <span className="font-semibold text-white">{link.query_index}:{link.query_token}</span>
                <span className="mx-2 text-slate-600">attends to</span>
                <span className="font-semibold text-sky-300">{link.key_index}:{link.key_token}</span>
              </div>
              <div className="rounded-full border border-emerald-400/20 bg-emerald-400/10 px-3 py-1 text-xs font-semibold text-emerald-300">
                {(link.weight * 100).toFixed(1)}%
              </div>
            </div>
            <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-slate-800">
              <div
                className="h-full rounded-full bg-gradient-to-r from-sky-400 to-emerald-400"
                style={{ width: `${Math.min(100, link.weight * 100)}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function PredictionPanel({ predictions = [] }) {
  return (
    <div className="studio-card rounded-3xl p-5">
      <p className="section-kicker">Next token distribution</p>
      <h3 className="mt-2 text-xl font-semibold text-white">Prediction Probe</h3>

      <div className="mt-5 space-y-3">
        {predictions.map((item, index) => (
          <div key={`${item.token}-${index}`}>
            <div className="mb-1 flex items-center justify-between text-sm">
              <span className="font-semibold text-white">{item.token}</span>
              <span className="text-slate-500">{(item.probability * 100).toFixed(2)}%</span>
            </div>
            <div className="h-1.5 overflow-hidden rounded-full bg-slate-800">
              <div
                className="h-full rounded-full bg-gradient-to-r from-sky-400 to-emerald-400"
                style={{ width: `${Math.max(3, item.probability * 100)}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function PositionalGrid({ matrix = [] }) {
  if (!matrix.length) return null

  const maxAbs = Math.max(...matrix.flat().map((value) => Math.abs(value)), 1)

  return (
    <div className="studio-card rounded-3xl p-5">
      <div className="mb-5">
        <p className="section-kicker">Position signal</p>
        <h3 className="text-xl font-semibold text-white">Sinusoidal Positional Encoding</h3>
      </div>
      <div className="overflow-auto rounded-2xl border border-slate-800 bg-slate-950 p-3">
        <div
          className="grid gap-1"
          style={{ gridTemplateColumns: `repeat(${matrix[0].length}, minmax(10px, 1fr))` }}
        >
          {matrix.map((row, rowIndex) =>
            row.map((value, colIndex) => {
              const intensity = Math.abs(value) / maxAbs
              const positive = value >= 0
              return (
                <div
                  key={`${rowIndex}-${colIndex}`}
                  className="aspect-square min-h-3 rounded-sm border border-slate-800"
                  title={`position ${rowIndex}, dim ${colIndex}: ${value}`}
                  style={{
                    background: positive
                      ? `rgba(14, 165, 233, ${0.08 + intensity * 0.85})`
                      : `rgba(16, 185, 129, ${0.08 + intensity * 0.85})`,
                  }}
                />
              )
            }),
          )}
        </div>
      </div>
    </div>
  )
}

export default function App() {
  const [activeTab, setActiveTab] = useState('workbench')
  const [config, setConfig] = useState(null)
  const [health, setHealth] = useState(null)
  const [analysisInput, setAnalysisInput] = useState(defaultAnalysis)
  const [analysis, setAnalysis] = useState(null)
  const [positional, setPositional] = useState(null)
  const [generation, setGeneration] = useState({
    seed_text: 'attention',
    max_new_tokens: 120,
    temperature: 0.8,
    mode: 'claude',
  })
  const [generatedText, setGeneratedText] = useState('')
  const [generationNote, setGenerationNote] = useState('')
  const [claudeStatus, setClaudeStatus] = useState(null)
  const [attentionExplanation, setAttentionExplanation] = useState('')
  const [loading, setLoading] = useState(false)
  const [apiError, setApiError] = useState('')

  useEffect(() => {
    async function boot() {
      try {
        const [healthData, configData, positionalData, claudeData] = await Promise.all([
          getHealth(),
          getConfig(),
          getPositionalEncoding(48, 48),
          testClaude(),
        ])
        setHealth(healthData)
        setConfig(configData)
        setPositional(positionalData)
        setClaudeStatus(claudeData)
      } catch {
        setApiError('Backend is not connected. Start it with: python -m uvicorn backend.main:app --reload --port 8000')
      }
    }

    boot()
    runAnalysis(defaultAnalysis)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  async function runAnalysis(payload = analysisInput) {
    setLoading(true)
    setApiError('')
    try {
      const data = await analyzeAttention(payload)
      setAnalysis(data)
    } catch {
      setApiError('Could not analyze attention. Make sure the FastAPI backend is running on port 8000.')
    } finally {
      setLoading(false)
    }
  }

  async function runGeneration() {
    setLoading(true)
    setApiError('')
    try {
      const data = await generateText(generation)
      setGeneratedText(data.generated_text)
      setGenerationNote(data.note || '')
    } catch {
      setApiError('Could not generate text. Make sure the backend is running.')
    } finally {
      setLoading(false)
    }
  }

  async function runClaudeExplanation() {
    setLoading(true)
    setApiError('')
    try {
      const data = await explainAttention({
        input_text: analysisInput.text,
        layer: analysisInput.layer,
        head: analysisInput.head,
        causal_mask: analysisInput.causal_mask,
        analysis_mode: analysisInput.analysis_mode,
      })
      setAttentionExplanation(data.explanation)
    } catch {
      setApiError('Could not generate Claude explanation. Check the backend and API key.')
    } finally {
      setLoading(false)
    }
  }

  const layerOptions = useMemo(() => {
    const count = config?.num_layers ?? 6
    return Array.from({ length: count }, (_, index) => index)
  }, [config])

  const headOptions = useMemo(() => {
    const count = config?.num_heads ?? 6
    return Array.from({ length: count }, (_, index) => index)
  }, [config])

  return (
    <main className="min-h-screen bg-slate-950 text-white">
      <div className="mx-auto max-w-7xl px-5 py-8 md:px-8">
        <header className="mb-10">
          <nav className="mb-8 flex flex-col gap-4 border-b border-slate-800 pb-6 md:flex-row md:items-center md:justify-between">
            <div>
              <p className="text-sm font-medium text-slate-500">Created by</p>
              <p className="text-lg font-semibold tracking-tight text-white">Omer Faizan</p>
            </div>
            <div className="flex items-center gap-2 rounded-full border border-slate-800 bg-slate-900 px-4 py-2 text-sm text-slate-400">
              <span className={`h-2 w-2 rounded-full ${health?.status === 'ok' ? 'bg-emerald-400' : 'bg-rose-400'}`} />
              {health?.status === 'ok' ? 'Backend online' : 'Backend offline'}
            </div>
          </nav>

          <div className="grid gap-6 lg:grid-cols-[1.2fr_0.8fr] lg:items-stretch">
            <div className="hero-panel rounded-[2rem] p-8 md:p-10">
              <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-slate-700 bg-slate-900 px-4 py-2 text-sm font-semibold text-slate-300">
                <Brain size={18} />
                Raw PyTorch Transformer Studio
              </div>

              <h1 className="max-w-4xl text-5xl font-semibold leading-[0.98] tracking-tight text-white md:text-7xl">
                AttentionForge
                <span className="block text-slate-400">Studio</span>
              </h1>

              <p className="mt-6 max-w-2xl text-lg leading-8 text-slate-400">
                A professional workbench for inspecting multi-head attention, causal masking,
                positional encoding, and generation behavior in a raw PyTorch Transformer.
              </p>

              <div className="mt-8 grid gap-3 sm:grid-cols-2">
                <div className="rounded-2xl border border-slate-800 bg-slate-950/70 p-4">
                  <p className="text-sm font-semibold text-white">No black-box APIs</p>
                  <p className="mt-1 text-sm text-slate-500">No Hugging Face for internals; raw mode now uses a trained local checkpoint.</p>
                </div>
                <div className="rounded-2xl border border-slate-800 bg-slate-950/70 p-4">
                  <p className="text-sm font-semibold text-white">Visual diagnostics</p>
                  <p className="mt-1 text-sm text-slate-500">Attention maps, token routes, and probes.</p>
                </div>
              </div>
            </div>

            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-1">
              <MetricCard label="Parameters" value={config?.parameter_count?.toLocaleString() ?? '—'} hint={config?.checkpoint_loaded ? "Trained checkpoint loaded" : "Checkpoint not loaded"} />
              <MetricCard label="Depth" value={`${config?.num_layers ?? 3} layers`} hint="Decoder-style Transformer" />
              <MetricCard label="Attention" value={`${config?.num_heads ?? 4} heads`} hint={`d_model ${config?.d_model ?? 64}`} />
            </div>
          </div>

          <div className="mt-6 flex flex-wrap gap-3">
            <TabButton active={activeTab === 'workbench'} onClick={() => setActiveTab('workbench')}>Workbench</TabButton>
            <TabButton active={activeTab === 'masking'} onClick={() => setActiveTab('masking')}>Masking</TabButton>
            <TabButton active={activeTab === 'position'} onClick={() => setActiveTab('position')}>Position Encoding</TabButton>
            <TabButton active={activeTab === 'generate'} onClick={() => setActiveTab('generate')}>Model Output</TabButton>
            <TabButton active={activeTab === 'architecture'} onClick={() => setActiveTab('architecture')}>Architecture</TabButton>
          </div>
        </header>

        {apiError ? (
          <div className="mb-8 rounded-2xl border border-rose-400/30 bg-rose-500/10 p-5 text-rose-100">
            {apiError}
          </div>
        ) : null}

        {activeTab === 'workbench' && (
          <section>
            <SectionHeader eyebrow="attention workbench" title="Inspect a selected head" icon={Eye}>
              Type a sentence, select a layer and head, and inspect how the raw PyTorch model routes information between tokens.
            </SectionHeader>

            <div className="grid gap-6 lg:grid-cols-[0.82fr_1.18fr]">
              <div className="space-y-5">
                <div className="studio-card rounded-3xl p-5">
                  <label className="text-sm font-semibold text-slate-300">Input text</label>
                  <textarea
                    value={analysisInput.text}
                    onChange={(event) => setAnalysisInput({ ...analysisInput, text: event.target.value })}
                    className="mt-3 h-32 w-full resize-none rounded-2xl border border-slate-800 bg-slate-950 p-4 text-white outline-none ring-sky-400/20 transition focus:ring-4"
                  />

                  <div className="mt-4 grid grid-cols-2 gap-4">
                    <label className="text-sm font-semibold text-slate-300">
                      Layer
                      <select
                        value={analysisInput.layer}
                        onChange={(event) => setAnalysisInput({ ...analysisInput, layer: Number(event.target.value) })}
                        className="mt-2 w-full rounded-2xl border border-slate-800 bg-slate-950 p-3 text-white outline-none"
                      >
                        {layerOptions.map((layer) => <option key={layer} value={layer}>Layer {layer}</option>)}
                      </select>
                    </label>

                    <label className="text-sm font-semibold text-slate-300">
                      Head
                      <select
                        value={analysisInput.head}
                        onChange={(event) => setAnalysisInput({ ...analysisInput, head: Number(event.target.value) })}
                        className="mt-2 w-full rounded-2xl border border-slate-800 bg-slate-950 p-3 text-white outline-none"
                      >
                        {headOptions.map((head) => <option key={head} value={head}>Head {head}</option>)}
                      </select>
                    </label>
                  </div>

                  <label className="mt-4 block text-sm font-semibold text-slate-300">
                    Analysis mode
                    <select
                      value={analysisInput.analysis_mode}
                      onChange={(event) => setAnalysisInput({ ...analysisInput, analysis_mode: event.target.value })}
                      className="mt-2 w-full rounded-2xl border border-slate-800 bg-slate-950 p-3 text-white outline-none"
                    >
                      <option value="guided">Guided interpretability mode</option>
                      <option value="raw">Raw PyTorch attention mode</option>
                    </select>
                    <span className="mt-2 block text-xs leading-5 text-slate-500">
                      Guided mode is best for portfolio demos. Raw mode shows actual trained checkpoint weights.
                    </span>
                  </label>

                  <label className="mt-4 flex cursor-pointer items-center gap-3 rounded-2xl border border-slate-800 bg-slate-950 p-3">
                    <input
                      type="checkbox"
                      checked={analysisInput.causal_mask}
                      onChange={(event) => setAnalysisInput({ ...analysisInput, causal_mask: event.target.checked })}
                      className="h-5 w-5"
                    />
                    <span>
                      <span className="block font-semibold text-white">Causal mask</span>
                      <span className="text-sm text-slate-500">Prevent future-token attention</span>
                    </span>
                  </label>

                  <button
                    onClick={() => runAnalysis()}
                    disabled={loading}
                    className="mt-5 flex w-full items-center justify-center gap-2 rounded-2xl bg-white px-5 py-4 font-semibold text-slate-950 transition hover:bg-slate-200 disabled:opacity-60"
                  >
                    <TerminalSquare size={20} />
                    {loading ? 'Running model...' : 'Run Analysis'}
                  </button>
                </div>

                <div className="studio-card rounded-3xl p-5">
                  <p className="section-kicker">Tokenized input</p>
                  <div className="mt-4">
                    <TokenRibbon tokens={analysis?.tokens ?? []} />
                  </div>
                </div>

                <PredictionPanel predictions={analysis?.next_token_predictions ?? []} />
              </div>

              <div className="space-y-5">
                <Heatmap
                  matrix={analysis?.attention_matrix ?? []}
                  tokens={analysis?.tokens ?? []}
                  title={`Layer ${analysis?.layer ?? 0} · Head ${analysis?.head ?? 0} · ${analysis?.analysis_mode ?? 'guided'} mode`}
                />

                {analysis?.source_note ? (
                  <div className="rounded-2xl border border-slate-800 bg-slate-950 p-4 text-sm leading-6 text-slate-400">
                    {analysis.source_note}
                  </div>
                ) : null}

                <div className="studio-card rounded-3xl p-5">
                  <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                    <div>
                      <p className="section-kicker">Claude Narrator</p>
                      <h3 className="mt-2 text-xl font-semibold text-white">Explain this attention view</h3>
                    </div>
                    <button
                      onClick={runClaudeExplanation}
                      disabled={loading}
                      className="rounded-2xl bg-white px-4 py-3 text-sm font-semibold text-slate-950 transition hover:bg-slate-200 disabled:opacity-60"
                    >
                      Explain with Claude
                    </button>
                  </div>
                  <div className="mt-4 whitespace-pre-wrap rounded-2xl border border-slate-800 bg-slate-950 p-4 text-sm leading-7 text-slate-300">
                    {attentionExplanation || 'Click the button to generate a clean natural-language explanation of the selected layer, head, mask, and top token routes.'}
                  </div>
                </div>

                <Connections links={analysis?.top_connections ?? []} />
              </div>
            </div>
          </section>
        )}

        {activeTab === 'masking' && (
          <section>
            <SectionHeader eyebrow="causal masking" title="The guardrail behind next-token prediction" icon={ShieldCheck}>
              Causal masking makes a decoder model honest by stopping it from reading tokens that appear later in the sequence.
            </SectionHeader>

            <div className="grid gap-6 md:grid-cols-2">
              <div className="studio-card rounded-3xl p-6">
                <h3 className="text-2xl font-semibold text-white">With mask</h3>
                <p className="mt-3 leading-7 text-slate-400">
                  Each token can attend only to itself and previous tokens. This is the correct setup for autoregressive language modeling.
                </p>
                <div className="mt-6 rounded-2xl border border-slate-800 bg-slate-950 p-5 font-mono text-sm text-sky-300">
                  scores.masked_fill(mask == 0, -inf)
                </div>
              </div>

              <div className="studio-card rounded-3xl p-6">
                <h3 className="text-2xl font-semibold text-white">Without mask</h3>
                <p className="mt-3 leading-7 text-slate-400">
                  Tokens can read future positions. This can be useful for inspection, but it leaks information for next-token prediction.
                </p>
                <div className="mt-6 rounded-2xl border border-slate-800 bg-slate-950 p-5 font-mono text-sm text-emerald-300">
                  softmax(QKᵀ / sqrt(d_head))
                </div>
              </div>
            </div>

            <div className="mt-6">
              <Heatmap
                matrix={analysis?.attention_matrix ?? []}
                tokens={analysis?.tokens ?? []}
                title={analysis?.causal_mask ? 'Current view: masked attention' : 'Current view: unmasked attention'}
              />
            </div>
          </section>
        )}

        {activeTab === 'position' && (
          <section>
            <SectionHeader eyebrow="positional encoding" title="How order is injected into parallel attention" icon={Activity}>
              Self-attention processes tokens in parallel, so the model needs an explicit signal that tells it where each token appears.
            </SectionHeader>
            <PositionalGrid matrix={positional?.matrix ?? []} />
          </section>
        )}

        {activeTab === 'generate' && (
          <section>
            <SectionHeader eyebrow="model output" title="Generate with Claude or the trained raw Transformer" icon={Play}>
              Claude mode gives polished explanations. Trained raw Transformer mode now uses a trained local PyTorch checkpoint, so it produces focused project-specific output instead of random text.
            </SectionHeader>

            <div className="grid gap-6 lg:grid-cols-[0.78fr_1.22fr]">
              <div className="studio-card rounded-3xl p-5">
                <label className="text-sm font-semibold text-slate-300">Input / seed text</label>
                <input
                  value={generation.seed_text}
                  onChange={(event) => setGeneration({ ...generation, seed_text: event.target.value })}
                  className="mt-3 w-full rounded-2xl border border-slate-800 bg-slate-950 p-4 text-white outline-none ring-sky-400/20 transition focus:ring-4"
                />

                <label className="mt-5 block text-sm font-semibold text-slate-300">
                  Mode
                  <select
                    value={generation.mode}
                    onChange={(event) => setGeneration({ ...generation, mode: event.target.value })}
                    className="mt-2 w-full rounded-2xl border border-slate-800 bg-slate-950 p-3 text-white outline-none"
                  >
                    <option value="claude">Claude API mode</option>
                    <option value="coherent">Offline demo explanation</option>
                    <option value="raw">Trained raw Transformer mode</option>
                  </select>
                </label>

                <div className={`mt-5 rounded-2xl border p-4 text-sm leading-6 ${
                  claudeStatus?.claude_configured
                    ? 'border-emerald-400/30 bg-emerald-400/10 text-emerald-100'
                    : 'border-amber-400/30 bg-amber-400/10 text-amber-100'
                }`}>
                  <div className="font-semibold">
                    Claude status: {claudeStatus?.claude_configured ? 'Connected' : 'Not connected'}
                  </div>
                  <div className="mt-1 text-xs opacity-90">
                    Model: {claudeStatus?.claude_model || 'not detected'}
                  </div>
                  {!claudeStatus?.claude_configured ? (
                    <div className="mt-2 text-xs">
                      The backend cannot see ANTHROPIC_API_KEY, so Claude API mode will use offline fallback.
                    </div>
                  ) : null}
                </div>

                <label className="mt-5 block text-sm font-semibold text-slate-300">
                  Output length: {generation.max_new_tokens}
                  <input
                    type="range"
                    min="20"
                    max="260"
                    step="20"
                    value={generation.max_new_tokens}
                    onChange={(event) => setGeneration({ ...generation, max_new_tokens: Number(event.target.value) })}
                    className="mt-3 w-full"
                  />
                </label>

                <label className="mt-5 block text-sm font-semibold text-slate-300">
                  Temperature: {generation.temperature}
                  <input
                    type="range"
                    min="0.1"
                    max="2"
                    step="0.1"
                    value={generation.temperature}
                    onChange={(event) => setGeneration({ ...generation, temperature: Number(event.target.value) })}
                    className="mt-3 w-full"
                  />
                </label>

                <button
                  onClick={runGeneration}
                  disabled={loading}
                  className="mt-6 flex w-full items-center justify-center gap-2 rounded-2xl bg-white px-5 py-4 font-semibold text-slate-950 transition hover:bg-slate-200 disabled:opacity-60"
                >
                  <Sparkles size={20} />
                  {loading ? 'Running model...' : 'Run output'}
                </button>
              </div>

              <div className="studio-card rounded-3xl p-5">
                <p className="section-kicker">Model output</p>
                {generationNote ? (
                  <div className={`mt-4 rounded-2xl border p-4 text-sm leading-6 ${
                    generationNote.toLowerCase().includes('unavailable') || generationNote.toLowerCase().includes('fallback')
                      ? 'border-amber-400/30 bg-amber-400/10 text-amber-100'
                      : 'border-sky-400/30 bg-sky-400/10 text-sky-100'
                  }`}>
                    {generationNote}
                  </div>
                ) : null}

                <pre className="mt-4 min-h-80 whitespace-pre-wrap rounded-2xl border border-slate-800 bg-slate-950 p-5 font-mono text-sm leading-7 text-slate-200">
                  {generatedText || 'Output will appear here. Use Claude for best language quality, or trained raw Transformer mode to show the local PyTorch checkpoint working.'}
                </pre>
              </div>
            </div>
          </section>
        )}

        {activeTab === 'architecture' && (
          <section>
            <SectionHeader eyebrow="architecture" title="A serious mini Transformer, not a baby demo" icon={Cpu}>
              The studio uses a trained lightweight Transformer checkpoint for raw mode, plus Claude as an optional explanation layer.
            </SectionHeader>

            <div className="grid gap-6 md:grid-cols-3">
              {[
                {
                  icon: Layers3,
                  title: `${config?.num_layers ?? 3} trained Transformer layers`,
                  text: 'Stacked decoder blocks refine token representations layer by layer and load from the trained local checkpoint.',
                },
                {
                  icon: GitBranch,
                  title: `${config?.num_heads ?? 4} attention heads`,
                  text: 'Each head creates a different relationship view over the input sequence.',
                },
                {
                  icon: Brain,
                  title: 'Raw PyTorch internals',
                  text: 'Queries, keys, values, masks, and attention scores are implemented manually.',
                },
              ].map((item) => (
                <div key={item.title} className="studio-card rounded-3xl p-6">
                  <item.icon className="mb-5 text-sky-300" size={32} />
                  <h3 className="text-2xl font-semibold text-white">{item.title}</h3>
                  <p className="mt-3 leading-7 text-slate-400">{item.text}</p>
                </div>
              ))}
            </div>

            <div className="mt-6 studio-card rounded-3xl p-6">
              <p className="section-kicker">Core formula</p>
              <div className="mt-4 rounded-2xl border border-slate-800 bg-slate-950 p-6 font-mono text-sky-300">
                Attention(Q, K, V) = softmax(QKᵀ / √d_head) V
              </div>
              <p className="mt-4 leading-7 text-slate-400">
                AttentionForge Studio exposes this mechanism through API diagnostics and visual heatmaps,
                making the model easier to explain in a portfolio, interview, or technical blog.
              </p>
            </div>
          </section>
        )}

        <footer className="mt-16 border-t border-slate-800 py-8 text-sm text-slate-500">
          <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
            <p>AttentionForge Studio · Made by Omer Faizan</p>
            <p>Raw PyTorch · FastAPI · React · No Hugging Face</p>
          </div>
        </footer>
      </div>
    </main>
  )
}
