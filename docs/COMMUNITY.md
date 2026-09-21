# Making JevLoop easy to join

A checklist, not a policy. Every item says why it matters **here** and how you would know it is done. Ordered by leverage, not by effort.

The bar this list is trying to meet: someone who has never talked to us clones the repo, runs one command, reads one file, and can tell whether a change of theirs would be welcome — without asking.

---

## Done

| | |
|---|---|
| ✅ | `README.md` + `README.zh-CN.md`, linked both ways |
| ✅ | [`CONTRIBUTING.md`](../CONTRIBUTING.md) — what we merge, what we close, where the work is |
| ✅ | CI that runs the four local commands plus two checks nobody can run by hand (`dist/`↔`src/` export parity, offline smoke) |
| ✅ | "Not affiliated with TypeSafe AI" stated in both READMEs |
| ✅ | PR template and bug-report template — the bar is visible at the moment someone opens one |
| ✅ | `good first issue` and `help wanted` labels exist |

## 1. Open the first issues — the highest-leverage thing on this list

**Why:** the labels are already there and `open_issues_count` is **0**. Everything else on this page is infrastructure; this is the only item that turns it into an invitation. A newcomer reads CONTRIBUTING, follows "where the work actually is", and finds an empty tracker.

**Done when:** three to five issues exist, each with a named seam or an already-decided shape, and at least one carries `good first issue`.

Candidates, all of them things the repo has already decided it wants:

- **Split the `Tool` seam** (`tools.ts` → `act.ts` + `act-local.ts`). See §10. Put the intended shape in the issue body — see "deliberately not doing" below for why it does not belong in a new doc.
- **`web/app.js` / `web/app.css`** — the front-end half of the 待拆 queue (§12). The most independent piece of work in the repo.
- **`--offline` for the demo** — see item 2.
- **The remaining Chinese in failure paths** — see item 4. This is the ideal `good first issue`: mechanical, verifiable, and it requires no architectural decisions.

## 2. Make the first command work in every environment

**Why:** `npm run demo` is the first thing the README tells anyone to run, in a repo whose headline promise is "no key, no network".

**Evidence (2026-09-21):**

```
$ npm run demo -- --rule
  decision  : rule-judge
  generator : http(deepseek-flash)   ← still resolved from .env
```

`prefer` is a single value, so `--rule` forces the decision backend and nothing else. With a `.env` present and its LLM host unreachable, the demo exits with a raw stack trace and `UND_ERR_CONNECT_TIMEOUT` — there is no flag that asks for a fully offline run, and `cd`-ing somewhere without a `.env` is the only workaround.

This is by design at the library level: the generation path has no fallback, and `runAgent`'s JSDoc says the caller must catch it. The demo is a caller and does not.

**Done when:** one flag produces the fresh-clone run from any directory.

## 3. Publish to npm

**Why:** `import { Decider } from 'jevloop'` in the README does not resolve — `registry.npmjs.org/jevloop` is a 404. Today the honest instruction is `npm install github:zjunlp/JevLoop`, which is already noted in the README, but "the package in the docs does not exist" is the kind of detail that makes a project look abandoned.

**Done when:** `npm install jevloop` works and the README's install line is the plain one.

## 4. Finish the English in failure paths

**Why:** the happy path is fully English (verified: zero Chinese lines in a normal run). The failure paths are not: the `reason:` strings in `src/decisions.ts`, `src/policy.ts` and `src/decide.ts`, and the budget warnings in `src/budget.ts`, are Chinese and surface inside trace lines and `onWarn`.

**Done when:** a run that escalates, stops or blows a budget prints English too.

## 5. Write down how fast we answer

**Why:** the single biggest factor in whether an early contributor comes back is whether their first PR got a response. That is a sentence, not code, and saying it is what makes it a commitment rather than an accident.

**Done when:** CONTRIBUTING states a response expectation and the project keeps it. If the expectation cannot be met, say the honest smaller number instead.

## 6. Check Windows

**Why:** untested, and untested is a guess. `src/tools.ts` is path-locked to `cwd`, and the demo writes raw ANSI escapes.

**Done when:** either someone has run `npm run demo` and `npm test` on Windows and the README says so, or the README says Windows is untested.

---

## Later — only once there are several contributors

- **`CODE_OF_CONDUCT.md`.** Its job is to exist before the first conflict, not before the first contributor. Cheap, and it is noticed by its absence exactly when it is needed.
- **`CHANGELOG.md`.** Low value for a 0.1.0 with no users. When it does earn its place, the thing that needs versioning is the **`DECISION.md` format** — that is the public contract others will write against — not the internal file layout.

---

## Deliberately not doing

- **Porting `DESIGN-layers-2026-09-21.md` into this repo.** Its P1 plan has largely landed (`types.ts` was split into `vocab*.ts`, `frame.ts` was extracted, `verify:layers` became the `layers` rule in `npm run check`). Its diagnostics name `src/types.ts` and a 318-line `src/agent.ts` — neither is true now. Its §5 (the two-kernel question) was settled by §8.7.1. Its §8 lists `npm run verify:port` as an acceptance criterion, and that script no longer exists. Most of all, **the layer table is machine-enforced in `scripts/check.ts`**; a prose copy would drift, which is the specific failure this repo has worked hardest to eliminate. The one unlanded piece — the `Tool` seam, P1-c — goes in its issue, where it will not be mistaken for current architecture.
- ~~**Fixing the two `../docs/DESIGN-layers-2026-09-21.md` links in `AGENTS.md` by copying the target in.**~~ **Done, the other way.** They were dropped rather than redirected, as decided here: for anyone who clones only this repo they were dead, and a copy of a plan that has moved on is worse than no copy. That file is now `docs/CODE-STYLE.md` (it was never about agents), and the `Tool` seam those links pointed at is described in place — target shape, not a reference to a document in another repository.
- **A `docs/` tree for its own sake.** This file is the first thing that needs to live here.
