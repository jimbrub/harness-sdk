/**
 * Prompt-caching gap audit: runs the default harness for a few multi-cycle turns per model and reports how much
 * of each model call's input was read from cache.
 *
 * Unlike `caching.test.ts` (one Bedrock model, no tools, pass/fail), this runs the real harness (environment and
 * todos injection, memory, session, tools, effort) across providers and prints a table, so a provider whose
 * `strategy: 'auto'` caches little or nothing stands out.
 *
 * Setup, from a checkout of the commit under test (harness-ts is not an npm workspace, so link it to this
 * checkout's SDK; the harness imports the SDK's built `dist`, so rebuild after every checkout):
 *   npm ci                                   # repo root
 *   (cd strands-ts && npm run build)
 *   cd harness-ts && npm ci && rm -rf node_modules/@strands-agents/sdk && ln -s ../../../strands-ts node_modules/@strands-agents/sdk
 *   cp /path/to/cache-audit.ts test/integration/
 *
 * Usage (from harness-ts/):
 *   npx tsx test/integration/cache-audit.ts                       # default model matrix
 *   npx tsx test/integration/cache-audit.ts bedrock/us.amazon.nova-pro-v1:0 openai/gpt-5.4
 *   CACHE_AUDIT_OUT=/tmp/cache-audit npx tsx test/integration/cache-audit.ts
 *   CACHE_AUDIT_CACHING=off npx tsx test/integration/cache-audit.ts   # caching-off baseline
 *   CACHE_AUDIT_PLUGINS=default npx tsx test/integration/cache-audit.ts   # the harness's own builtinPlugins default
 *
 * Credentials: Bedrock models use the default AWS credential chain (us-east-1 unless AWS_REGION is set);
 * `anthropic/`, `openai/` and `google/` read ANTHROPIC_API_KEY, OPENAI_API_KEY and GEMINI_API_KEY (or Vertex
 * settings). A model whose credentials are missing reports its error in the table and the run continues.
 *
 * The report header records the commit, the SDK the harness resolved, and the plugins, so a run can be matched to
 * the code it measured. Only main-loop model calls are measured (`agent.metrics` cycles); auxiliary calls such as
 * memory distillation or summarization are not.
 */

import { execFileSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { mkdirSync, mkdtempSync, realpathSync, statSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

import type { Agent } from '@strands-agents/sdk'

import { createHarness } from '../../src/agent.js'

// `CACHE_AUDIT_CACHING=off` runs the same turns with the harness's caching turned off, the baseline for the cost
// benchmark. Providers that cache implicitly (OpenAI, Gemini, Bedrock OpenAI GPT) still cache server-side.
const CACHING_OFF = process.env.CACHE_AUDIT_CACHING === 'off'

// `todos` left the harness's default plugins in #5974. The turns below exercise it, so it stays on by default to
// keep runs comparable with earlier audits; `CACHE_AUDIT_PLUGINS=default` uses the harness's own default instead.
const PLUGINS_SETTING = process.env.CACHE_AUDIT_PLUGINS ?? 'todos,environment'
const BUILTIN_PLUGINS =
  PLUGINS_SETTING === 'default'
    ? undefined
    : (PLUGINS_SETTING.split(',')
        .map((name) => name.trim())
        .filter(Boolean) as Array<'todos' | 'environment'>)

const DEFAULT_MODELS = [
  'bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0',
  'bedrock/global.anthropic.claude-sonnet-4-6',
  'bedrock/us.amazon.nova-pro-v1:0',
  'bedrock/us.amazon.nova-lite-v1:0',
  'bedrock/us.amazon.nova-micro-v1:0',
  'bedrock/us.amazon.nova-2-lite-v1:0',
  'bedrock/us.xai.grok-4.7',
  'bedrock/openai.gpt-oss-120b-1:0',
  'bedrock-mantle/openai.gpt-5.6-luna',
  'bedrock/us.openai.gpt-5.6-luna',
  'bedrock/us.openai.gpt-6-luna',
  'bedrock-mantle/openai.gpt-oss-120b',
  'anthropic/claude-haiku-4-5',
  'anthropic/claude-sonnet-4-6',
  'openai/gpt-5.6',
  'openai/gpt-5.4',
  'google/gemini-3.5-flash',
  'google/gemini-3-flash-preview',
]

// Each turn asks for tool calls so a turn spans several model calls, and the injectors (environment, todos,
// memory) have something new to fold in on every turn. Turns 3 and 4 need the `todos` plugin.
const TURNS = [
  'Remember for later: my favorite color is teal. Create a file notes.txt containing the line "alpha", then read it back and tell me its contents.',
  'Append the line "beta" to notes.txt, then show me the whole file.',
  'Add a todo "review notes", mark it in progress, then list the files in the current directory.',
  'What is my favorite color? Answer in one word, then mark the todo completed.',
]

interface Usage {
  inputTokens: number
  outputTokens: number
  totalTokens: number
  cacheReadInputTokens?: number
  cacheWriteInputTokens?: number
}

interface CallRow {
  model: string
  turn: number
  call: number
  input: number
  cacheRead: number
  cacheWrite: number
  hit: number
}

/** Where the prompt sent to the model stopped matching the previous call's prompt. */
interface PrefixTrace {
  model: string
  call: number
  messages: number
  /** `system`, `tools`, `msg[i]`, or `—` when the previous prompt is an unchanged prefix of this one. */
  breaksAt: string
  /** For a `system` or `msg[i]` break: the text around the first differing character, previous → current. */
  detail?: string
}

interface ModelSummary {
  model: string
  calls: number
  hitAfterFirstCall: number
  zeroReadCalls: number
  warnings: string[]
  error?: string
}

/** Total prompt size of one call; providers differ on whether `inputTokens` already includes cache tokens. */
function totalInput(usage: Usage): number {
  const read = usage.cacheReadInputTokens ?? 0
  const write = usage.cacheWriteInputTokens ?? 0
  return usage.inputTokens + usage.outputTokens === usage.totalTokens
    ? usage.inputTokens
    : usage.inputTokens + read + write
}

function lastInvocationCalls(agent: Agent): Usage[] {
  const invocations = (agent as unknown as { metrics: { agentInvocations: { cycles: { usage: Usage }[] }[] } }).metrics
    .agentInvocations
  return (invocations.at(-1)?.cycles ?? []).map((cycle) => cycle.usage)
}

/** Captures caching-related console output (SDK warnings) emitted while `run` executes. */
async function captureWarnings<T>(warnings: string[], run: () => Promise<T>): Promise<T> {
  const original = { warn: console.warn, error: console.error }
  const capture =
    (forward: (...args: unknown[]) => void) =>
    (...args: unknown[]): void => {
      const text = args.map(String).join(' ')
      if (/cach|strategy/i.test(text)) warnings.push(text)
      forward(...args)
    }
  console.warn = capture(original.warn)
  console.error = capture(original.error)
  try {
    return await run()
  } finally {
    console.warn = original.warn
    console.error = original.error
  }
}

const hash = (value: unknown): string =>
  createHash('sha1')
    .update(JSON.stringify(value) ?? '')
    .digest('hex')
    .slice(0, 12)

const asJson = (value: unknown): unknown =>
  value !== null && typeof value === 'object' && 'toJSON' in value ? (value as { toJSON(): unknown }).toJSON() : value

interface Snapshot {
  system: string
  tools: string
  messages: string[]
}

function firstDifference(previous: string, current: string): string {
  let index = 0
  while (index < previous.length && previous[index] === current[index]) index++
  const window = (text: string): string => JSON.stringify(text.slice(Math.max(0, index - 40), index + 80))
  return `@${index}: ${window(previous)} → ${window(current)}`
}

function breakPoint(previous: Snapshot, current: Snapshot): Pick<PrefixTrace, 'breaksAt' | 'detail'> {
  if (previous.system !== current.system)
    return { breaksAt: 'system', detail: firstDifference(previous.system, current.system) }
  if (previous.tools !== current.tools) return { breaksAt: 'tools' }
  const index = previous.messages.findIndex((message, i) => message !== current.messages[i])
  if (index === -1) return { breaksAt: '—' }
  return { breaksAt: `msg[${index}]`, detail: firstDifference(previous.messages[index]!, current.messages[index]!) }
}

/**
 * Records what the agent hands the model on every call: the model's own `stream`, after the agent's injection and
 * the model proxy, so a prompt prefix that changes between calls (and breaks implicit caching) shows up here.
 */
function tracePrefixes(agent: Agent, model: string, traces: PrefixTrace[]): void {
  const target = agent.model as unknown as {
    stream: (messages: unknown[], options?: Record<string, unknown>) => unknown
  }
  const original = target.stream.bind(target)
  let previous: Snapshot | undefined
  let call = 0
  target.stream = (messages, options) => {
    const systemPrompt = options?.systemPrompt
    const current: Snapshot = {
      system: typeof systemPrompt === 'string' ? systemPrompt : JSON.stringify(systemPrompt ?? null),
      tools: hash(options?.toolSpecs ?? null),
      messages: messages.map((message) => JSON.stringify(asJson(message)) ?? ''),
    }
    call++
    traces.push({
      model,
      call,
      messages: messages.length,
      ...(previous ? breakPoint(previous, current) : { breaksAt: '(first)' }),
    })
    previous = current
    return original(messages, options)
  }
}

async function auditModel(model: string): Promise<{ rows: CallRow[]; traces: PrefixTrace[]; summary: ModelSummary }> {
  const previous = process.cwd()
  process.chdir(mkdtempSync(join(tmpdir(), 'strands-cache-audit-')))
  const rows: CallRow[] = []
  const traces: PrefixTrace[] = []
  // Collected outside the try so a model that errors part-way still reports its warnings.
  const warnings: string[] = []
  try {
    await captureWarnings(warnings, async () => {
      const agent = await createHarness({
        model,
        ...(CACHING_OFF ? { caching: false } : {}),
        ...(BUILTIN_PLUGINS ? { builtinPlugins: BUILTIN_PLUGINS } : {}),
      })
      tracePrefixes(agent, model, traces)
      for (const [index, prompt] of TURNS.entries()) {
        await agent.invoke(prompt)
        lastInvocationCalls(agent).forEach((usage, call) => {
          const input = totalInput(usage)
          const cacheRead = usage.cacheReadInputTokens ?? 0
          rows.push({
            model,
            turn: index + 1,
            call: call + 1,
            input,
            cacheRead,
            cacheWrite: usage.cacheWriteInputTokens ?? 0,
            hit: input === 0 ? 0 : cacheRead / input,
          })
        })
      }
    })
    const later = rows.slice(1)
    const laterInput = later.reduce((sum, row) => sum + row.input, 0)
    const laterRead = later.reduce((sum, row) => sum + row.cacheRead, 0)
    return {
      rows,
      traces,
      summary: {
        model,
        calls: rows.length,
        hitAfterFirstCall: laterInput === 0 ? 0 : laterRead / laterInput,
        zeroReadCalls: later.filter((row) => row.cacheRead === 0).length,
        warnings: [...new Set(warnings)],
      },
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    return {
      rows,
      traces,
      summary: {
        model,
        calls: rows.length,
        hitAfterFirstCall: 0,
        zeroReadCalls: 0,
        warnings: [...new Set(warnings)],
        error: message,
      },
    }
  } finally {
    process.chdir(previous)
  }
}

interface RunInfo {
  commit: string
  sdk: string
  sdkBuiltAt: string
  plugins: string
  caching: string
  startedAt: string
}

function git(args: string[]): string {
  try {
    return execFileSync('git', args, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim()
  } catch {
    return 'unknown'
  }
}

/** Which commit and which SDK build this run measured; a stale SDK `dist` silently measures old code. */
function runInfo(): RunInfo {
  const dirty = git(['status', '--porcelain', '--untracked-files=no']) === '' ? '' : ' (uncommitted changes)'
  const sdkEntry = fileURLToPath(import.meta.resolve('@strands-agents/sdk'))
  return {
    commit: `${git(['rev-parse', 'HEAD'])}${dirty}`,
    sdk: realpathSync(dirname(sdkEntry)),
    sdkBuiltAt: statSync(sdkEntry).mtime.toISOString(),
    plugins: BUILTIN_PLUGINS ? BUILTIN_PLUGINS.join(', ') : 'harness default',
    caching: CACHING_OFF ? 'off' : 'harness default (auto)',
    startedAt: new Date().toISOString(),
  }
}

const pct = (value: number): string => `${(value * 100).toFixed(0)}%`

/** Escapes a markdown table cell; SDK log lines contain `|`. */
const cell = (text: string): string => text.replace(/\|/g, '\\|').replace(/\n/g, ' ')

function render(info: RunInfo, summaries: ModelSummary[], rows: CallRow[], traces: PrefixTrace[]): string {
  const lines = [
    '## Run',
    '',
    `- commit: \`${info.commit}\``,
    `- SDK: \`${info.sdk}\` (built ${info.sdkBuiltAt})`,
    `- builtin plugins: ${info.plugins}; caching: ${info.caching}`,
    `- started: ${info.startedAt}`,
    '',
    '## Summary (calls after the first; the first call can only write)',
    '',
    '| model | calls | cache hit | calls with 0 reads | warnings | error |',
    '|---|---|---|---|---|---|',
    ...summaries.map(
      (s) =>
        `| ${s.model} | ${s.calls} | ${pct(s.hitAfterFirstCall)} | ${s.zeroReadCalls} | ${cell(s.warnings.join('<br>')) || '—'} | ${cell(s.error ?? '—')} |`
    ),
    '',
    '## Per call',
    '',
    '| model | turn | call | input | cache read | cache write | hit |',
    '|---|---|---|---|---|---|---|',
    ...rows.map(
      (r) => `| ${r.model} | ${r.turn} | ${r.call} | ${r.input} | ${r.cacheRead} | ${r.cacheWrite} | ${pct(r.hit)} |`
    ),
    '',
    '## Prefix changes (what the agent sent the model, compared with the previous call)',
    '',
    '`—` means the previous prompt is an unchanged prefix of this one. `system`/`tools`/`msg[i]` names the first part',
    'that changed; anything before the last message changing breaks implicit (provider-side) prefix caching.',
    '',
    '| model | call | messages | breaks at | detail |',
    '|---|---|---|---|---|',
    ...traces.map((t) => `| ${t.model} | ${t.call} | ${t.messages} | ${t.breaksAt} | ${cell(t.detail ?? '—')} |`),
  ]
  return lines.join('\n')
}

async function main(): Promise<void> {
  const models = process.argv.slice(2).length > 0 ? process.argv.slice(2) : DEFAULT_MODELS
  const outDir = process.env.CACHE_AUDIT_OUT ?? join(tmpdir(), `strands-cache-audit-${Date.now()}`)
  mkdirSync(outDir, { recursive: true })
  const info = runInfo()
  process.stderr.write(`commit ${info.commit}\nsdk ${info.sdk} (built ${info.sdkBuiltAt})\n`)

  const summaries: ModelSummary[] = []
  const rows: CallRow[] = []
  const traces: PrefixTrace[] = []
  for (const model of models) {
    process.stderr.write(`auditing ${model} ...\n`)
    const result = await auditModel(model)
    summaries.push(result.summary)
    rows.push(...result.rows)
    traces.push(...result.traces)
    // Written after every model so a crash or timeout keeps what finished.
    writeFileSync(join(outDir, 'results.json'), JSON.stringify({ run: info, turns: TURNS, summaries, rows, traces }, null, 2))
    writeFileSync(join(outDir, 'REPORT.md'), render(info, summaries, rows, traces))
  }

  process.stdout.write(`${render(info, summaries, rows, traces)}\n\nwritten to ${outDir}\n`)
}

await main()
