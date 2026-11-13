/**
 * Gaps 19 and 20, on Bedrock Mantle's Responses API. Needs credentials that can call Mantle: the earlier attempt
 * failed with a 403 on `bedrock-mantle:CreateInference`, so this probe is the one that needs the elevated role.
 *
 * Gap 19: does Mantle accept `prompt_cache_breakpoint` on `input_image` / `input_file` for GPT-6.x? The Bedrock docs
 * list those block types only for GPT-5.6, so the SDK may be placing a breakpoint the API rejects.
 *
 * Gap 20: the Responses adapter sends a document as `input_file` with a filename that has no extension, which
 * Mantle rejected with a 400 for every model. Part B sends the same document named `order` and `order.pdf` to
 * confirm the cause and that an extension fixes it - the evidence PR 4 needs before changing the adapter.
 *
 * Run from `strands-ts/` after copying this directory to `strands-ts/cache-probes/` (see README.md).
 */

import { Buffer } from 'node:buffer'

import { Agent, TextBlock } from '@strands-agents/sdk'
import type { ContentBlock } from '@strands-agents/sdk'

import { OpenAIModel } from '../src/models/openai/model.js'
import { DocumentBlock, ImageBlock } from '../src/types/media.js'
import { describeError, header, report, verdict } from './_shared.ts'

const PNG = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAeUlEQVR4nO3PQQkAMAzAwGqqfwGTNRF7HINABFzm7H7dcEEDWtCAFjSgBQ1oQQNa0IAWNKAFDWhBA1rQgBY0oAUNaEEDWtCAFjSgBQ1oQQNa0IAWNKAFDWhBA1rQgBY0oAUNaEEDWtCAFjSgBQ1oQQNa0IAWNKAFj10ThMEPv3x7AAAAAABJRU5ErkJggg==',
  'base64'
)
const PDF = Buffer.from(
  'JVBERi0xLjQKMSAwIG9iajw8L1R5cGUvQ2F0YWxvZy9QYWdlcyAyIDAgUj4+ZW5kb2JqCjIgMCBvYmo8PC9UeXBlL1BhZ2VzL0tpZHNbMyAwIFJdL0NvdW50IDE+PmVuZG9iagozIDAgb2JqPDwvVHlwZS9QYWdlL1BhcmVudCAyIDAgUi9NZWRpYUJveFswIDAgMzAwIDE0NF0vQ29udGVudHMgNCAwIFIvUmVzb3VyY2VzPDwvRm9udDw8L0YxIDUgMCBSPj4+Pj4+ZW5kb2JqCjQgMCBvYmo8PC9MZW5ndGggNDQ+PnN0cmVhbQpCVCAvRjEgMTggVGYgMjAgMTAwIFRkIChPcmRlciA0MiBzaGlwcGVkKSBUaiBFVAplbmRzdHJlYW0gZW5kb2JqCjUgMCBvYmo8PC9UeXBlL0ZvbnQvU3VidHlwZS9UeXBlMS9CYXNlRm9udC9IZWx2ZXRpY2E+PmVuZG9iagp0cmFpbGVyPDwvUm9vdCAxIDAgUj4+CiUlRU9G',
  'base64'
)

const MODELS = (process.env.PROBE_MODELS ?? 'openai.gpt-5.6-luna,openai.gpt-6-luna,openai.gpt-6.1-sol')
  .split(',')
  .map((id) => id.trim())
  .filter(Boolean)

const REGION = process.env.AWS_REGION ?? 'us-east-1'

/** Part types that carried a breakpoint, and filenames sent, captured from the last Responses request. */
let sentBreakpoints: string[] = []
let sentFilenames: string[] = []

const realFetch = globalThis.fetch
globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
  if (typeof init?.body === 'string' && init.body.includes('"input"')) {
    try {
      const body = JSON.parse(init.body) as {
        input?: Array<{ content?: Array<{ type: string; filename?: string; prompt_cache_breakpoint?: unknown }> }>
      }
      const parts = (body.input ?? []).flatMap((item) => (Array.isArray(item.content) ? item.content : []))
      sentBreakpoints = parts.filter((part) => part.prompt_cache_breakpoint !== undefined).map((part) => part.type)
      sentFilenames = parts.flatMap((part) => (part.filename === undefined ? [] : [part.filename]))
    } catch {
      // A body we cannot parse is not worth failing the probe over; the request still goes through.
    }
  }
  return realFetch(input, init)
}) as typeof globalThis.fetch

/**
 * Sends one turn and reports whether Mantle accepted it, plus what the adapter put on the wire.
 *
 * @param modelId - Mantle model id
 * @param content - Content blocks for the single user message
 * @returns A line for the report
 */
async function send(modelId: string, content: ContentBlock[]): Promise<string> {
  sentBreakpoints = []
  sentFilenames = []
  const model = new OpenAIModel({
    modelId,
    bedrockMantleConfig: { region: REGION },
    cacheConfig: {},
    maxTokens: 150,
  })
  const agent = new Agent({ model, printer: false, systemPrompt: 'Answer in five words or fewer.' })
  try {
    await agent.invoke(content)
    return `ACCEPTED | breakpoints on: ${sentBreakpoints.join(',') || 'none'} | filenames: ${sentFilenames.join(',') || 'none'}`
  } catch (error) {
    return `REJECTED | breakpoints on: ${sentBreakpoints.join(',') || 'none'} | filenames: ${sentFilenames.join(',') || 'none'} | ${describeError(error).slice(0, 300)}`
  }
}

const image = (): ContentBlock => new ImageBlock({ format: 'png', source: { bytes: new Uint8Array(PNG) } })
const pdf = (name: string): ContentBlock =>
  new DocumentBlock({ name, format: 'pdf', source: { bytes: new Uint8Array(PDF) } })

async function run(): Promise<void> {
  header('04 responses-media (gaps 19, 20) | needs Mantle access', MODELS)

  report('--- Part A, gap 19: breakpoint placement on media ---')
  for (const modelId of MODELS) {
    for (const [label, content] of [
      ['text only', [new TextBlock('Name one colour.')]],
      ['ends in image', [new TextBlock('Describe the image.'), image()]],
      ['ends in pdf', [new TextBlock('Describe the document.'), pdf('order.pdf')]],
      ['ends in image then pdf', [new TextBlock('Describe both.'), image(), pdf('order.pdf')]],
    ] as Array<[string, ContentBlock[]]>) {
      report(`${modelId} | A | ${label} | ${await send(modelId, content)}`)
    }
  }

  report('--- Part B, gap 20: input_file filename extension ---')
  for (const modelId of MODELS.slice(0, 2)) {
    for (const [label, name] of [
      ['name without extension', 'order'],
      ['name with .pdf', 'order.pdf'],
    ] as Array<[string, string]>) {
      report(`${modelId} | B | ${label} | ${await send(modelId, [new TextBlock('Describe the document.'), pdf(name)])}`)
    }
  }

  verdict(
    'gap 19',
    'see Part A: say for each model whether a request ending in an image or a PDF was accepted, and which part types carried prompt_cache_breakpoint'
  )
  verdict(
    'gap 20',
    'see Part B: say whether the extensionless filename was rejected and the .pdf one accepted, and quote the 400 verbatim'
  )
}

await run()
