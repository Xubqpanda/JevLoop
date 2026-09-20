# nanojev

**The agent loop where decisions don't cost a model call.**

Every fork in a normal agent loop — *should I act? which tool? is this safe? did it work? am I done? can I ship this?* — is answered by a full LLM call. None of those are generation. They're picks, scores and yes/no answers.

nanojev routes them to a decision model ([Jev](https://typesafe.ai) / [Laya](https://github.com/NandaKishorM/laya)) and keeps the LLM for the one thing only it can do: **writing**.

```
$ npm run demo

  ── loop trace ──────────────────────────────────────────
  判定放行：list_dir（auto）
  判定放行：read_file（auto）

  ── 记账 ────────────────────────────────────────────────
  判定   11 次   46.4ms（均 4.2ms）
  模型    1 次   600.5ms

  判定 : 模型 = 11.0 : 1      判定耗时只占 7.2%
```

Zero dependencies. Zero build step. Runs offline with no API key.

---

## The problem

Take a task that needs two tool calls. A conventional agent burns a model call on each of these:

| Question the loop asks | Conventional agent | nanojev |
|---|---|---|
| Do I need to act yet? | LLM call | decision |
| Which tool? | LLM call | decision |
| Is this call safe? | LLM call, or nothing at all | decision |
| Did it work? | LLM call | decision |
| Am I done? | `max_iter` counter | decision |
| Can I ship this answer? | **nothing** | decision |

You were paying generation prices for decisions. A decision is one forward pass over a fixed candidate set — no tokens generated, nothing to parse, ~10–40 ms on a GPU.

## Quick start

Needs **Node ≥ 22.6** (it runs TypeScript directly, no build).

```bash
git clone https://github.com/Xubqpanda/nanojev
cd nanojev
npm run demo
```

That's it. No `npm install`, no API key, no network — the demo falls back to a deterministic rule judge so the whole loop runs offline.

**Use a real decision model:**

```bash
npm run demo -- --laya     # local Laya sidecar on :7789 (open weights, free)
npm run demo -- --jev      # official Jev API (needs TYPESAFE_API_KEY)
```

## How it works

```
 step ─┬─ loop.needsTool   ↗ do I need to act?  ──no──▶ generate
       │
       ├─ loop.pickTool    ↗ which tool?  (options rebuilt every step)
       │
       ├─ loop.gradeRisk   ↗ how dangerous is this?  ──▶ ask a human
       │
       ├─ [ tool runs ]     ← the only place with real side effects
       │
       ├─ loop.stepOk      ↗ did it work?
       │
       └─ loop.isDone      ↗ am I done?  ──no──▶ next step
                            │
                            ▼
                       [ LLM generates ]   ← the only expensive call
                            │
                       loop.canDeliver  ↗ is this shippable?
```

**All six decision points live in one file: [`src/decisions.ts`](src/decisions.ts).** If you read one file in this repo, read that one — it's the whole idea.

### A decision is three things

```ts
export const pickTool = defineDecision({
  id: "loop.pickTool",

  // ① Project the agent state into a BOUNDED decision frame.
  //    This caps what the model can judge: what isn't in the
  //    frame cannot be decided.
  state: (ctx: AgentCtx) => ({
    task: clip(ctx.task, 400),
    files_known: (ctx.files ?? []).slice(0, 20),
    recent: (ctx.history ?? []).slice(-3).map(h => `${h.tool}(${h.input}) → ${clip(h.result, 120)}`),
  }),

  // ② Typed questions. Answered in ONE forward pass.
  questions: (ctx: AgentCtx) => ({
    tool: choice("Which tool should the agent call next?", toolsFor(ctx)),
  }),

  // ③ Policy: answers → action. Pure code. No model involved.
  policy: [
    { when: gte("tool", 0.6), action: "call" },
    { action: "escalate", reason: "not confident enough — hand back, don't guess" },
  ],
});
```

Three primitives, taken straight from the Jev wire protocol:

| Primitive | Answer | Used for |
|---|---|---|
| `noul` | P(true), 0–1 | **gate** — allow / block |
| `choice` | one option + per-option probability | **route** — which path |
| `score` | expected level on an ordered scale | **grade** — how bad |

### Two things worth stealing

**Rebuild the options every step.** A fixed action list makes the model pick something that no longer applies — `write_file` should not still be a candidate after you've written the file. That's why `questions` can be a function of the context.

**Never let a probability bypass authorisation.** Risk gating is a hard rule, not a threshold:

```ts
policy: [
  // irreversible ⇒ explicit authorisation. No confidence score overrides this.
  { when: scoreGte("risk", 2), action: "ask_human" },
  // the model's own read is a SECOND, independent gate
  { when: probGte("needs_auth", 0.5), action: "ask_human" },
  { action: "auto" },
]
```

A decision model may decide *whether to ask a human*. It must never decide *whether to skip authorisation*.

## Accounting

The `Meter` is not a nice-to-have — it's the point. Every run ends with the number that justifies the architecture:

```
判定 : 模型 = 11.0 : 1      判定耗时只占 7.2%
```

Decisions and model calls are counted separately, with separate latency. If that ratio isn't high for your workload, nanojev is the wrong tool — and you should find out immediately rather than after a bill.

## Bring your own backends

**Decision backend** — anything that answers `{state, questions} → {answers}`:

```ts
import { Decider, HttpProvider, FallbackProvider, MockProvider } from "nanojev";

const decider = new Decider({
  provider: new FallbackProvider([
    new HttpProvider({ baseUrl: "https://api.typesafe.ai", apiKey: process.env.TYPESAFE_API_KEY, name: "jev" }),
    new HttpProvider({ baseUrl: "http://127.0.0.1:7789", name: "laya" }),
    new MockProvider(),   // never fails
  ]),
});
```

**Generation backend** — anything that turns a prompt into text:

```ts
import { HttpGenerator } from "nanojev";
// any OpenAI-compatible /chat/completions endpoint
new HttpGenerator({ baseUrl: "https://api.openai.com/v1", apiKey, model: "gpt-5" });
new HttpGenerator({ baseUrl: "http://localhost:11434/v1", model: "qwen3" });  // ollama
```

Swapping either one touches exactly one file. The loop and the decision specs don't move.

## What this is not

- **Not a replacement for an LLM.** Drafting, coding and summarising still need one.
- **Not "zero hallucination".** A decision model can't return an answer outside the type you asked for, but the answer can still be wrong. That's what the confidence is for.
- **Not benchmarked yet.** The `11:1` above is from the bundled demo. A real comparison against a conventional agent on the same task is the obvious next step — and it isn't done.
- **The demo's judge is a rule table, not a model.** `examples/rule-judge.ts` is a deterministic stand-in so `npm run demo` works with no key and no network. Real numbers need `--laya` or `--jev`. The file is clearly marked and gets deleted the moment you have a real backend.
- **Not production-hardened.** Tool sandboxing covers path escape only. Read `src/tools.ts` before pointing it at anything you care about.

## Layout

```
src/
  types.ts       Question / Answer / Decision — the whole vocabulary
  decisions.ts   ★ all six of the agent's judgements, one file
  decide.ts      the six steps of one decision
  policy.ts      answers → action (pure code, unit-testable)
  provider.ts    Jev / Laya / Mock — swap by baseUrl
  meter.ts       ★ decisions vs model calls
  agent.ts       ★ the loop
  tools.ts       list / read / write, path-locked to cwd
  llm.ts         the one place that generates
examples/
  demo.ts        runs offline
  rule-judge.ts  deterministic stand-in for a decision model
```

## License

Apache-2.0
