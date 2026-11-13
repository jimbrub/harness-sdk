/**
 * Gap 15 / Nova open question 1: in a multi-turn chat (no tools), which conversation cache-point placement reads
 * most from cache on Nova?
 *
 * Nova reads an entry only when a cache point repeats the previous call's position, and it does not look back for
 * an earlier one. #6082 ships `fixed`: the point stays on the last user message that has text, which in a tool loop
 * is the prompt that started it. Plain chat is the open case, because every turn adds a user message with text, so
 * `fixed` moves every turn and may never read.
 *
 * Strategies, each run as its own conversation so they never share entries:
 *   none    no cache point (baseline, shows what implicit caching alone gives)
 *   moving  one point at the end of the newest user message (what the SDK did before #6082)
 *   fixed   one point after the first user message's text, never moved (what #6082 ships for tool loops)
 *   pair    points after the previous and the newest user message's text (the "keep the last two" proposal)
 *
 * Decides: whether PR 9 keeps the fixed point for chat or adds the rolling pair.
 */

import { assistant, call, header, pad, report, text, user, verdict } from './_shared.ts'
import type { ContentBlock, Message, SystemContentBlock } from '@aws-sdk/client-bedrock-runtime'

const MODELS = (process.env.PROBE_MODELS ?? 'us.amazon.nova-pro-v1:0,us.amazon.nova-lite-v1:0,us.amazon.nova-2-lite-v1:0')
  .split(',')
  .map((id) => id.trim())
  .filter(Boolean)

const TURNS = [
  'Name one fact about the Pacific Ocean.',
  'Now one fact about the Atlantic.',
  'Which of the two is deeper?',
  'How much deeper, roughly?',
  'Name a trench in the deeper one.',
  'How was that trench formed?',
]

type Strategy = 'none' | 'moving' | 'fixed' | 'pair'
const STRATEGIES: Strategy[] = ['none', 'moving', 'fixed', 'pair']
const CP: ContentBlock = { cachePoint: { type: 'default' } }

/**
 * Builds the message list for a turn, placing cache points per strategy.
 *
 * @param history - Alternating user/assistant messages from earlier turns, without cache points
 * @param strategy - Placement under test
 * @returns Messages to send, with cache points inserted
 */
function place(history: Message[], strategy: Strategy): Message[] {
  const messages = history.map((message) => ({ ...message, content: [...(message.content ?? [])] }))
  const userIdxs = messages.flatMap((message, index) => (message.role === 'user' ? [index] : []))
  const lastUser = userIdxs.at(-1)
  const previousUser = userIdxs.at(-2)

  if (strategy === 'moving' && lastUser !== undefined) {
    messages[lastUser]!.content.push(CP)
  }
  if (strategy === 'fixed' && userIdxs[0] !== undefined) {
    messages[userIdxs[0]]!.content.push(CP)
  }
  if (strategy === 'pair') {
    // Descending, so the first splice does not shift the second index.
    for (const index of [lastUser, previousUser].filter((value): value is number => value !== undefined)) {
      messages[index]!.content.push(CP)
    }
  }
  return messages
}

async function run(): Promise<void> {
  header('01 nova-placement (gap 15 / PR 9)', MODELS)
  const system: SystemContentBlock[] = [{ text: `You are a terse geography tutor.\n${pad(2200)}` }]

  for (const modelId of MODELS) {
    const totals: Record<Strategy, { read: number; reads: number; calls: number }> = {
      none: { read: 0, reads: 0, calls: 0 },
      moving: { read: 0, reads: 0, calls: 0 },
      fixed: { read: 0, reads: 0, calls: 0 },
      pair: { read: 0, reads: 0, calls: 0 },
    }

    for (const strategy of STRATEGIES) {
      const history: Message[] = []
      const perCall: string[] = []

      for (const turn of TURNS) {
        history.push(user(text(turn)))
        const result = await call(modelId, { messages: place(history, strategy), system })
        if (!result.ok) {
          report(`${modelId} | ${strategy} | turn ${perCall.length + 1} | REJECTED | ${result.error}`)
          break
        }
        perCall.push(`r${result.read}/w${result.write}`)
        // Later calls are the only ones that can read; the first always writes.
        if (perCall.length > 1) {
          totals[strategy].calls += 1
          totals[strategy].read += result.read
          if (result.read > 0) totals[strategy].reads += 1
        }
        history.push(assistant(text(result.reply.slice(0, 200) || 'ok')))
      }
      report(`${modelId} | ${strategy} | ${perCall.join(' ')}`)
    }

    const ranked = STRATEGIES.map((strategy) => ({ strategy, ...totals[strategy] })).sort(
      (left, right) => right.read - left.read
    )
    const summary = ranked
      .map((entry) => `${entry.strategy} ${entry.reads}/${entry.calls} calls read, ${entry.read} tok`)
      .join('; ')
    report(`${modelId} | SUMMARY | ${summary}`)
    verdict(
      `gap 15 ${modelId}`,
      `best placement for plain chat is '${ranked[0]!.strategy}' (${ranked[0]!.read} tokens read across later calls)`
    )
  }
}

await run()
