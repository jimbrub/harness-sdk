/**
 * Gaps 22 and 11: which Converse models cache that #6082 does not cover?
 *
 * #6082 treats OpenAI GPT (except gpt-oss) and xAI as implicitly cached, and the four measured Nova models as taking
 * cache points. Model cards list implicit caching for other families, and Kimi K3/K3.1 and GLM-5.3 were never
 * measured (the earlier attempt was throttled). Each model is classified by two calls with no cache point (does it
 * read anything on its own?) and one call with a point (is the point accepted?).
 *
 * Decides: which models PR 5 writes into shared/model-info.json, and with which mechanism.
 */

import { call, header, pad, report, text, user, verdict } from './_shared.ts'
import type { ContentBlock } from '@aws-sdk/client-bedrock-runtime'

const MODELS = (
  process.env.PROBE_MODELS ??
  [
    'us.moonshotai.kimi-k3',
    'us.moonshotai.kimi-k3.1',
    'zai.glm-5.3',
    'zai.glm-4.7',
    'us.deepseek.v3-v1:0',
    'us.meta.llama4-maverick-17b-instruct-v1:0',
    'mistral.mistral-large-3-675b-instruct',
    'openai.gpt-oss-safeguard-120b',
    'us.amazon.nova-premier-v1:0',
    'us.amazon.nova-2-pro-preview-20251202-v1:0',
  ].join(',')
)
  .split(',')
  .map((id) => id.trim())
  .filter(Boolean)

const CP: ContentBlock = { cachePoint: { type: 'default' } }
const PROMPT = (): ContentBlock => text(`Summarise these rules in one sentence.\n${pad(2400)}`)

/**
 * Classifies one model: implicit caching, cache points, or neither.
 *
 * @param modelId - Converse model id
 * @returns A one-line classification for the matrix
 */
async function classify(modelId: string): Promise<string> {
  const prompt = PROMPT()

  const first = await call(modelId, { messages: [user(prompt)] })
  if (!first.ok) return `UNREACHABLE | ${first.error}`

  const second = await call(modelId, { messages: [user(prompt)] })
  const implicit = second.ok && second.read > 0

  const withPoint = await call(modelId, { messages: [user(prompt, CP)] })
  const pointAccepted = withPoint.ok
  const pointRead = withPoint.ok ? withPoint.read : 0
  const pointWrite = withPoint.ok ? withPoint.write : 0

  const mechanism = implicit
    ? 'implicit'
    : pointAccepted && (pointWrite > 0 || pointRead > 0)
      ? 'cachePoints'
      : pointAccepted
        ? 'none (point accepted but nothing cached)'
        : 'none (point rejected)'

  const detail = [
    `no-point reads: call1 ${first.read}, call2 ${second.read}`,
    `with-point: ${pointAccepted ? `accepted, read ${pointRead}, write ${pointWrite}` : `rejected | ${withPoint.error}`}`,
  ].join(' | ')

  return `${mechanism} | ${detail}`
}

async function run(): Promise<void> {
  header('03 longtail-implicit (gaps 22, 11 -> PR 5 data)', MODELS)

  const lines: string[] = []
  for (const modelId of MODELS) {
    const result = await classify(modelId)
    report(`${modelId} | ${result}`)
    lines.push(`${modelId}: ${result.split(' | ')[0]}`)
  }

  verdict('gaps 22 and 11', lines.join('; '))
  report(
    'NOTE: a model that reads on call 2 with no cache point is implicitly cached and needs promptCaching "implicit"; one that only reads with a point needs "cachePoints"; one that reads neither way needs "unsupported".'
  )
}

await run()
