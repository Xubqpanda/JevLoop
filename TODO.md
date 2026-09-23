# TODO

JevLoop already demonstrates one thing: **the judgements in an agent loop can be
taken away from the generative model** — and that this can be declared, measured
and reproduced.

What it is not yet is a harness you would run something real on. This file is the
distance between those two sentences, ordered by what blocks the claim rather
than by difficulty.

If you want to take one of these on, **open an issue saying so first**. We will
scope it with you before you write code — cheaper for both of us than a large PR
that has to be reshaped. `CONTRIBUTING.md` says what we merge and what we close.

---

## 1 · The tool surface is four tools wide, so the decision space is too

**Why this blocks the claim.** Every judgement in this loop is shaped around four
tools — `list_dir`, `read_file`, `write_file`, `done` — and the most dangerous
action available is writing a file.

That leaves `grade_risk` close to untestable. **A risk ladder only means something
if there is something genuinely risky to climb it.** The breadth of the decision
space also decides how far "judgements can leave the model" can be verified at
all: four tools only ever demonstrate file operations, and that is a narrow claim
to build a paper on.

- [ ] Split the tool seam — interface from implementation (`act.ts` +
      `act-local.ts`). The intended shape is in `docs/CODE-STYLE.md` §10.
- [ ] Add tools with real consequences: shell execution, git operations.
- [ ] Give every level of the risk ladder something real to point at.

**Where:** `src/tools.ts` — 142 lines holding the interface, the implementation
and the side effects. **Size:** medium.

## 2 · Decision frames have no cache policy, so the cost argument is missing a leg

**Why this blocks the claim.** Cost is one of this project's two claims. Structural
prompt caching prices a repeated prefix at roughly 0.1×, and published
measurements of a harness with a deliberate cache shape report **99.9% of prompt
tokens served as cache reads**. It is the single largest cost lever there is, and
we use none of it.

Decision frames are unusually well suited to it: `task` is invariant, `history` is
append-only, and every frame already carries a digest. Without this, any
cost-per-task comparison we publish is missing its largest term.

- [ ] Split the frame into a byte-stable prefix and a tail rebuilt each step.
- [ ] Record the cache hit rate — in `usage.json` and in the UI's accounting panel.
- [ ] Verify: repeated runs of the same task should show a stable hit rate.

**Where:** `src/frame.ts`, `src/provider-http.ts`. **Size:** medium.

## 3 · It does not run long, and compaction is not a clean slate

**Why this blocks the claim.** The loop finishes after a few dozen steps. Folding
solves "do not overflow this turn"; it does not solve either of the two things a
long task actually needs.

**Context reset.** Compaction preserves continuity — it does not give the agent a
clean start. Long tasks need the second thing, and right now we only have the
first.

**Memory across sessions.** When a session ends, what it learned is gone. There is
no memory module in `src/`.

- [ ] Context reset: after a reset the agent continues the work rather than
      restarting it.
- [ ] Cross-session memory: a later session can reach an earlier one's conclusions.
- [ ] Make "what must survive" declarative and checkable — the way decision frames
      already are.

**Where:** `src/context.ts` and `src/conversation.ts` (both budgets exist).
**Size:** large.

## 4 · Internal hygiene

None of this makes the project worse. All of it makes the next change slower.

- [ ] The five files with a decided seam: `src/agent.ts` (916 lines),
      `src/context.ts` (447), `src/decisiondoc.ts`, `web/app.js`, `web/app.css`.
      Split along the seam, never along a line count — `docs/CODE-STYLE.md` §12.
- [ ] Anything labelled `good first issue`.
- [ ] **Your own itch.** Something this loop does badly on your workload is more
      interesting to us than anything on this list.

---

## Already settled — do not re-litigate these

- **How a judgement leaves the generative model.** `DECISION.md` compiles to typed
  questions plus a policy; `FrameSpec` declares what each judgement sees.
- **What each judgement is allowed to look at.** Declared per node, with a digest,
  and the deliberately-excluded fields are written down with a reason.
- **What each judgement and each generation cost.** Counted separately, per task.
  Very few harnesses publish this at all, and none we know of split it this way.
- **How often each judgement is right.** `bench/` grades the seven nodes
  independently instead of reporting one accuracy number.

If you think one of these is wrong, that is a bug report and we want it — open an
issue with the command and the output. What we are not looking for is a
re-argument of the design without a measurement behind it.

## Before you open a PR

Four commands, all of them, on your machine:

```bash
npm run check        # style, layering direction, file focus, CSS scope
npm run typecheck
npm test             # the whole suite, offline
npm run demo         # the loop must run to completion, offline
```

A two-line fix with a failing test in front of it is a better PR than a large
feature without one. We cannot tell an AI-assisted change from any other and we
do not care which it is — only whether it runs.
