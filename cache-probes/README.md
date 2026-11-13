# Prompt-caching probes

Live probes that answer the open questions in the caching work. Each one prints `VERDICT <gap>: <answer>` so a run
can be collected by grepping for `VERDICT`. None of them write, modify or delete any AWS resource — they are all
inference calls.

Each probe exists to settle a decision, not to collect numbers:

| Probe | Question | Decides |
| --- | --- | --- |
| `01-nova-placement.ts` | On Nova in plain multi-turn chat, which conversation cache-point placement reads most? | whether PR 9 keeps the fixed point or adds the rolling pair |
| `02-cache-point-budget.ts` | What rejects when a request exceeds 4 cache points, and is the limit per request or per section? | PR 6's guard and its error wording |
| `03-longtail-implicit.ts` | Which Converse models cache that #6082 does not cover? | which models PR 5 writes into `shared/model-info.json` |
| `04-responses-media.ts` | Does Mantle accept a `prompt_cache_breakpoint` on `input_image` / `input_file`, and does an `input_file` filename need an extension? | gap 19, and validates PR 4's fix |

## Setup

The probes import the SDK, so they run from inside a checkout:

```bash
# from the repo root of the commit under test
npm ci
(cd strands-ts && npm run build)
cp -r cache-probes strands-ts/
cd strands-ts
```

Run each one from `strands-ts/`:

```bash
export AWS_REGION=us-east-1
export CACHE_PROBE_OUT=/tmp/cache-probes/results.txt      # every probe appends here

npx tsx cache-probes/01-nova-placement.ts
npx tsx cache-probes/02-cache-point-budget.ts
npx tsx cache-probes/03-longtail-implicit.ts
npx tsx cache-probes/04-responses-media.ts                # needs Mantle access
```

Override the models on any probe with `PROBE_MODELS=id1,id2`.

### Credentials

- `01`, `02`, `03` need only Bedrock Converse (`bedrock:InvokeModel`) in the target region.
- `04` needs **Mantle**: `bedrock-mantle:CreateInference`. A role without it fails with a 403, which is why this
  probe has never run.

## Already answered (us-east-1, 2026-11-13)

Do not re-run these unless you want a second sample; they are recorded here so the open list stays short.

**Gap 16 — the cache-point budget is per request, and one section may hold all 4.** Four points spread across
`toolConfig` + system + messages were accepted; five were rejected whether spread across sections or placed entirely
in messages; four in the system prompt alone were accepted. The rejection wording differs per family:

- Claude Haiku 4.5: `A maximum of 4 blocks with cache_control may be provided. Found 5.`
- Nova Lite: `Found more than maximum allowable 4 CachePoints. Please refer to the prompt caching documentation and update the input before trying again.`

**Gap 15 — in plain chat, the rolling pair beats the fixed point on 2 of 3 Nova models, and a single moving point
never reads.** Six-turn chat, reads on the five calls after the first:

| Model | none | moving | fixed | pair |
| --- | --- | --- | --- | --- |
| Nova Pro | 0/5 | 0/5 | 4/5, 15,080 tok | **5/5, 19,029 tok** |
| Nova Lite | 0/5 | 0/5 | **4/5, 15,018 tok** | 1/5, 3,803 tok |
| Nova 2 Lite | 0/5 | 0/5 | 4/5, 11,316 tok | **5/5, 14,372 tok** |

`moving` reading nothing confirms Nova has no lookback, and `none` reading nothing confirms Nova does not cache
implicitly. Nova Lite disagreeing with the other two is unexplained — see the repeat-sample note below.

**Gap 22 / 11, partially.** Four models cache nothing and reject a cache point with `You invoked an unsupported
model or your request did not allow prompt caching`: `zai.glm-4.7`, `us.meta.llama4-maverick-17b-instruct-v1:0`,
`mistral.mistral-large-3-675b-instruct`, `openai.gpt-oss-safeguard-120b`. That confirms #6082 is right to exclude
them. Six were unreachable from the current account and still need an answer:

| Model | Why it failed |
| --- | --- |
| `us.moonshotai.kimi-k3`, `us.moonshotai.kimi-k3.1` | `Too many tokens, please wait before trying again` (throttled) |
| `zai.glm-5.3` | on-demand throughput unsupported; needs an inference profile |
| `us.deepseek.v3-v1:0`, `us.amazon.nova-2-pro-preview-20251202-v1:0` | `The provided model identifier is invalid` — wrong id |
| `us.amazon.nova-premier-v1:0` | end of life; expected, no action |

Before re-running `03`, discover the real ids rather than guessing:

```bash
aws bedrock list-foundation-models --query 'modelSummaries[].modelId' --output text
aws bedrock list-inference-profiles --query 'inferenceProfileSummaries[].inferenceProfileId' --output text
```

## Repeat samples

Every number above is a single sample, and single samples have been unreliable: the same code gave Nova 36–79% on
2026-11-12 and 37–73% on 2026-11-13, and Grok 93–94% then 85–86%. Two audit runs 10–14 minutes apart also means no
warm-cache measurement exists anywhere in this work (gap 24).

So for any probe whose result decides something, run it **three times back to back** and report the spread. For the
harness audit specifically, run the three samples inside five minutes so at least one is warm. `cache-audit.ts` lives
in `harness-ts/`, not `strands-ts/`, and is not an npm workspace member - see the setup steps in its own header
comment (copy it from `cache-probes/cache-audit.ts` into `harness-ts/test/integration/`, `npm ci` there, symlink the
SDK). Once set up, from `harness-ts/`:

```bash
for sample in 1 2 3; do
  CACHE_AUDIT_OUT=/tmp/audit-warm-$sample npx tsx test/integration/cache-audit.ts
done
```

## Reporting

Return, for each probe: every `VERDICT` line, the per-model lines behind it, and any probe that failed with the
error verbatim rather than retried around. Also return the header line of each probe (region, nonce, timestamp) and
the commit the SDK was built from.
