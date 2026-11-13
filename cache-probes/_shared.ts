/**
 * Shared helpers for the prompt-caching probes. Raw Bedrock Converse only, no SDK placement logic, so each probe
 * measures the provider rather than our code.
 *
 * Run every probe from `strands-ts/` with `npx tsx ../cache-probes/<file>.ts` (see README.md).
 */

import { randomUUID } from 'node:crypto'
import { appendFileSync, mkdirSync } from 'node:fs'
import { dirname } from 'node:path'

import {
  BedrockRuntimeClient,
  ConverseCommand,
  type ContentBlock,
  type ConverseCommandInput,
  type ConverseCommandOutput,
  type Message,
  type SystemContentBlock,
  type Tool,
} from '@aws-sdk/client-bedrock-runtime'

export const REGION = process.env.AWS_REGION ?? 'us-east-1'
export const client = new BedrockRuntimeClient({ region: REGION })

/** Output file every probe appends its lines to, so one run produces one artifact. */
export const OUT = process.env.CACHE_PROBE_OUT ?? '/tmp/cache-probes/results.txt'

export const CP: ContentBlock = { cachePoint: { type: 'default' } }
export const SYS_CP: SystemContentBlock = { cachePoint: { type: 'default' } }

/** A run-unique nonce, so probes never read each other's cache entries. */
export const NONCE = randomUUID().slice(0, 8)

/** Roughly `tokens` tokens of stable, run-unique text (one line is about 12 tokens). */
export function pad(tokens: number, nonce = NONCE): string {
  const lines = Math.ceil(tokens / 12)
  return Array.from({ length: lines }, (_, index) => `Rule ${index} for run ${nonce}: answer briefly and precisely.`).join(
    '\n'
  )
}

export const user = (...content: ContentBlock[]): Message => ({ role: 'user', content })
export const assistant = (...content: ContentBlock[]): Message => ({ role: 'assistant', content })
export const text = (value: string): ContentBlock => ({ text: value })

export const TOOL: Tool = {
  toolSpec: {
    name: 'calculator',
    description: 'Evaluate an arithmetic expression.',
    inputSchema: { json: { type: 'object', properties: { expression: { type: 'string' } }, required: ['expression'] } },
  },
}

export interface CallResult {
  ok: boolean
  /** Input tokens read from cache on this call. */
  read: number
  /** Input tokens written to cache on this call. */
  write: number
  input: number
  /** First line of the model's reply, for turn-by-turn probes. */
  reply: string
  /** Provider error, verbatim and flattened, when the request was rejected. */
  error?: string
}

/** Flattens an error and its causes into one line, keeping the provider's wording intact. */
export function describeError(error: unknown): string {
  const parts: string[] = []
  let current: unknown = error
  while (current instanceof Error) {
    parts.push(current.message)
    current = current.cause
  }
  return parts.join(' <- ').replace(/\s+/g, ' ')
}

/** Sends one Converse call and reports its cache counters, or the provider's rejection. */
export async function call(modelId: string, request: Omit<ConverseCommandInput, 'modelId'>): Promise<CallResult> {
  try {
    const response: ConverseCommandOutput = await client.send(
      new ConverseCommand({ modelId, ...request, inferenceConfig: { maxTokens: 120, ...request.inferenceConfig } })
    )
    const usage = response.usage ?? {}
    return {
      ok: true,
      read: usage.cacheReadInputTokens ?? 0,
      write: usage.cacheWriteInputTokens ?? 0,
      input: usage.inputTokens ?? 0,
      reply: (response.output?.message?.content ?? []).map((block) => ('text' in block ? block.text : '')).join(' '),
    }
  } catch (error) {
    return { ok: false, read: 0, write: 0, input: 0, reply: '', error: describeError(error) }
  }
}

/** Prints a line to stdout and appends it to the probe artifact. */
export function report(line: string): void {
  process.stdout.write(`${line}\n`)
  mkdirSync(dirname(OUT), { recursive: true })
  appendFileSync(OUT, `${line}\n`)
}

/** Prints the single sentence that answers the probe's question. Grep `VERDICT` to collect them all. */
export function verdict(gap: string, answer: string): void {
  report(`VERDICT ${gap}: ${answer}`)
}

export function header(name: string, models: string[]): void {
  report('')
  report(`=== ${name} | region ${REGION} | nonce ${NONCE} | ${new Date().toISOString()} ===`)
  report(`models: ${models.join(', ')}`)
}
