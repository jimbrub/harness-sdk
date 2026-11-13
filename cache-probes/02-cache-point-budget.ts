/**
 * Gap 16 / PR 6: what exactly happens when a request carries more cache points than Converse allows, and is the
 * limit per request or per section?
 *
 * The SDK spends up to three points itself (toolConfig, system, the last user message), so a caller placing two
 * more can overflow. PR 6 needs to know the exact rejection wording to guard against, and whether the budget is
 * shared across toolConfig/system/messages or counted per section, before it can decide which point to drop.
 *
 * Decides: PR 6's guard — what to count, and what error to avoid.
 */

import { SYS_CP, TOOL, call, header, pad, report, text, user, verdict } from './_shared.ts'
import type { ContentBlock, Message, SystemContentBlock, Tool } from '@aws-sdk/client-bedrock-runtime'

const MODELS = (process.env.PROBE_MODELS ?? 'global.anthropic.claude-haiku-4-5-20251001-v1:0,us.amazon.nova-lite-v1:0')
  .split(',')
  .map((id) => id.trim())
  .filter(Boolean)

const CP: ContentBlock = { cachePoint: { type: 'default' } }

/** One padded text block, so every cached prefix clears Bedrock's minimum cacheable size. */
const block = (label: string): ContentBlock => text(`${label}\n${pad(1200)}`)
const sysBlock = (label: string): SystemContentBlock => ({ text: `${label}\n${pad(1200)}` })

interface Case {
  name: string
  /** What the 4-point limit predicts. */
  expect: string
  build(): {
    system?: SystemContentBlock[]
    toolConfig?: { tools: Tool[] }
    messages: Message[]
  }
}

const CASES: Case[] = [
  {
    name: '1 point in messages only',
    expect: 'accepted',
    build: () => ({ messages: [user(block('a'), CP)] }),
  },
  {
    name: '3 points: tools + system + messages',
    expect: 'accepted (what the SDK itself spends)',
    build: () => ({
      toolConfig: { tools: [TOOL, { cachePoint: { type: 'default' } } as unknown as Tool] },
      system: [sysBlock('s'), SYS_CP],
      messages: [user(block('a'), CP)],
    }),
  },
  {
    name: '4 points: tools + system + 2 in messages',
    expect: 'accepted, at the limit',
    build: () => ({
      toolConfig: { tools: [TOOL, { cachePoint: { type: 'default' } } as unknown as Tool] },
      system: [sysBlock('s'), SYS_CP],
      messages: [user(block('a'), CP, block('b'), CP)],
    }),
  },
  {
    name: '5 points: tools + system + 3 in messages',
    expect: 'rejected if the limit is per request',
    build: () => ({
      toolConfig: { tools: [TOOL, { cachePoint: { type: 'default' } } as unknown as Tool] },
      system: [sysBlock('s'), SYS_CP],
      messages: [user(block('a'), CP, block('b'), CP, block('c'), CP)],
    }),
  },
  {
    name: '5 points, all in messages',
    expect: 'rejected if the limit is per request; accepted if per section and the section allows 5',
    build: () => ({
      messages: [user(block('a'), CP, block('b'), CP, block('c'), CP, block('d'), CP, block('e'), CP)],
    }),
  },
  {
    name: '4 points, all in system',
    expect: 'shows whether a single section may hold 4',
    build: () => ({
      system: [sysBlock('s1'), SYS_CP, sysBlock('s2'), SYS_CP, sysBlock('s3'), SYS_CP, sysBlock('s4'), SYS_CP],
      messages: [user(block('a'))],
    }),
  },
]

async function run(): Promise<void> {
  header('02 cache-point-budget (gap 16 / PR 6)', MODELS)

  for (const modelId of MODELS) {
    const outcomes: string[] = []
    for (const probe of CASES) {
      const result = await call(modelId, probe.build())
      const outcome = result.ok ? `ACCEPTED (read ${result.read}, write ${result.write})` : `REJECTED | ${result.error}`
      outcomes.push(`${probe.name} -> ${result.ok ? 'accepted' : 'rejected'}`)
      report(`${modelId} | ${probe.name} | expect: ${probe.expect} | ${outcome}`)
    }
    report(`${modelId} | SUMMARY | ${outcomes.join('; ')}`)
  }

  verdict(
    'gap 16',
    'see the ACCEPTED/REJECTED lines above - quote the rejection message verbatim, and say whether 5 points failed in both layouts (per request) or only when spread across sections (per section)'
  )
}

await run()
