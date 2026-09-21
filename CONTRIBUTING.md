# Contributing to JevLoop

JevLoop is one idea with a small surface: route every judgement in an agent loop to a decision model, and keep the LLM for the one thing it can do — writing. The loop itself fits in a few files and you can read it in an afternoon. That is deliberate, and it is also why outside help matters: the interesting work is in the parts we have not written yet.

**[`AGENTS.md`](AGENTS.md) is the code rules** — layering, file focus, comment style, the CSS scoping rule — and `npm run check` enforces most of them mechanically. Read it before your first commit; it will save you a review round.

This file is the other half: what we want, what we close, and where the work actually is.

## Setup

Node ≥ 22.6 — the repo runs TypeScript directly, with no build step.

```bash
git clone https://github.com/zjunlp/JevLoop
cd JevLoop
npm install          # only for the TypeScript compiler; the tests and the demo need nothing
```

Nothing else: no API key and no model download. The test suite and the demo run entirely offline, which is also how CI runs them.

## The four commands that have to pass

CI runs these on every push and pull request, plus two more steps (below). Run them yourself first — a red CI on your first PR is a bad way to meet us.

```bash
npm run check        # style, layering direction, file focus, CSS scope
npm run typecheck
npm test             # the whole suite, offline
npm run demo         # the loop must run to completion, offline
```

CI also checks the two things that are easy to get wrong and impossible to notice:

- **`dist/` and `src/` must expose the same runtime surface.** The package entry points at `dist/`, but every local command exercises `src/`. That means the one surface a user actually gets is the only one nobody runs. A stale `dist/` once shipped seven missing modules and an orphan, with check, typecheck, test and demo all green.
- **The demo runs offline.** `DEEPSEEK_API_KEY= npm run demo -- --rule`.

## What we will merge

- **A fix with a reproduction.** The issue (or the PR body) contains the command that shows the bug and the output it prints. Not a description of the bug — the output.
- **A test that fails before the change and passes after.** If the change is not testable, say why in the PR.
- **A measurement.** If you claim something got faster, cheaper or more accurate, paste the command and the numbers. A number without a command behind it is a wish.
- **A smaller change rather than a bigger one.** A one-file, one-concern PR gets reviewed the same day. A sweeping one sits until someone has an hour.
- **A correction to something we got wrong.** That includes the README, `DECISION.md`, and the comments. If we claim something the code does not do, that is a bug and we want the issue.

## What we close

We would rather say this up front than waste your afternoon.

- **Code its author never ran.** We cannot tell an AI-assisted change from any other and we do not care which it is — only whether it runs. If `npm test` and `npm run demo` did not pass on your machine, the PR is closed. Bulk-generated PRs with no reproduction are the entire reason this section exists.
- **Prose-only "improvements".** Rewording the README without a code change or a reproducible defect is not a contribution, it is a diff.
- **Numbers nobody can reproduce.** Benchmarks without the harness, latencies without the command, and "10x faster" without the machine it ran on.
- **New dependencies.** Zero dependencies is a feature of this project, not an accident. A dependency PR needs to argue why the sixty lines it replaces cannot be written here.
- **Reformatting, renaming and drive-by refactors.** `npm run check` already decides formatting. Review bandwidth is the scarcest thing this project has.
- **A second copy of something that already exists.** If you find two implementations of one idea, say so in an issue and let us decide which one survives — do not consolidate them in a PR on your own initiative. The version being replaced is usually someone's work in progress.

None of this is about skill level. A two-line fix with a failing test in front of it is a better PR than a large feature without one.

## Where the work actually is

The repo keeps its own list of open work. These are the real ones — each already has something decided about it, which is what makes them startable:

| Work | Where it is described | Rough size |
|---|---|---|
| **Split the `Tool` seam.** `tools.ts` is the only capability seam in the repo with all three corners in one file, and it is the one that produces real side effects — the hardest to replace and the easiest to get wrong. The intended shape (`act.ts` for the interface, `act-local.ts` for the local-filesystem implementation) is sketched in `AGENTS.md` §10. | `AGENTS.md` §10 | medium |
| **The five files in the 待拆 queue** — `src/agent.ts`, `src/context.ts`, `src/decisiondoc.ts`, `web/app.js`, `web/app.css`. Each has a named seam; split along it, not along a line count. | `AGENTS.md` §12 | medium |
| **Anything in the issue tracker** labelled `good first issue` | GitHub issues | small |
| **Your own itch.** Something the loop does badly on your workload is more interesting to us than anything on this list. | — | — |

Every one of those needs the same three things: register a new `src/*.ts` in `scripts/check.ts`'s `LAYER` table in the same change, keep the module JSDoc honest about why the file cannot be split further, and run the four commands.

If you want to take one on, open an issue saying so and we will scope it with you before you write code. That is cheaper for both of us than a large PR that has to be reshaped.

## Reporting a bug

Open an issue with:

1. **What you ran** — the exact command, including flags.
2. **What it printed** — pasted, not summarised. If it is a crash, the stack.
3. **What you expected instead.**
4. **Your Node version** (`node --version`). Type stripping is v22.6+, and on older versions you get a `SyntaxError` that looks nothing like a version problem.

If the judgement itself was wrong rather than the code — the loop picked the wrong tool, the delivery gate rejected a good answer — say which decision it was and paste the decision frame you sent. **A wrong judgement is usually a frame problem, not a policy problem:** the model decides on what you project into the frame, and nothing else. The `state` projection of each decision is in [`src/decisions.ts`](src/decisions.ts).

## Commits

- One concern per commit, and change one repository at a time.
- **Commit messages in English.** Code comments and in-repo documents may be Chinese; messages are what an outsider reads first.
- Explain *why* in the body. What changed is in the diff.

## Licence

Apache-2.0. By contributing you agree your work is licensed under the same terms.
