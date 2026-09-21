# 跑实验的规范

**七个人共用一份。** 布局和命名在 `README.md`，数据集索引在 `dataset/MANIFEST.md`，
计划在伞仓库 `../../docs/PLAN-confidence-2026-09-21.md`。**这份只管「怎么跑、留下什么」。**

---

## 0. 一条不能破的

> ## 没有 log 的运行 = 没跑过。

**不是「最好留个日志」，是「拿不出 log 的数字不许进 `result/`」。**
理由在 §2。这一条是这份文件里唯一不能商量的。

---

## 1. 一次运行 = 一个目录

```
experiments/log/<dataset>/<arm>/<UTC 时间>-seed<N>/
├── cmd.txt        完整命令行 + 当时的环境变量（密钥按 §7 打掉）
├── meta.json      用了什么（字段见 §3.2）
├── trace.jsonl    逐判定一行（§3.3）—— ★ RQ2 的唯一来源
├── usage.json     逐调用的 token 与耗时（§3.4）
├── stdout.txt
├── stderr.txt
└── exit.json      退出码 + 是否跑完 + 中断原因（§3.5）
```

**例**：`log/bfcl-v4/jevloop/2026-09-22T03-14-07Z-seed0/`

**永不覆盖。** 重跑就是**新目录**。**跑挂的那次也留着** —— 失败率本身是指标。

---

## 2. 为什么 log 比 result 重要

| | `result/` | `log/` |
|---|---|---|
| 是什么 | **结论**：汇总表和图 | **证据**：这次到底发生了什么 |
| 错了怎么办 | 从 log **重算**就能修 | **没了就只能重跑** —— 而重跑**不是同一次** |
| 谁看 | 论文读者 | 审稿人、三个月后的我们、以及「这个数是怎么来的」 |

**`result/` 里每一个数字，都必须能指回一个 `run_id`。** 指不回去的数字就是传说。

**「任何主张都要有可复现的测量」**（伞仓库 `AGENTS.md` §8）的判据是
**「这句话，我能不能给出跑出它的那条命令？」** —— 那条命令就是 `cmd.txt`。

---

## 3. 每个文件里必须有什么

### 3.1 `cmd.txt`

完整命令行，**加上当时生效的环境变量**（`JEVLOOP_GATES` / `JEVLOOP_HOME` / 模型覆盖那类）。
**密钥按 §7 打掉。** 别人照着这一行必须能重跑。

### 3.2 `meta.json` —— `PLAN-*.md` §3.7 那一套，一个不少

```
run_id  started_at  finished_at  duration_ms  git_commit  dirty(是/否)
dataset  dataset_version  dataset_split  dump_version  task_count  sample_seed
arm  generator{id,version,provider}  decider{...}  thinking_budget
temperature  top_p  max_tokens  prompt_hash  decision_md_hash
region  cold_start(是/否)  max_steps  timeout_s  retry_policy
```

★ **`dirty` 必须记。** 用一个改了没提交的工作区跑出来的数字，
**别人永远复现不了** —— 它连自己都复现不了。

### 3.3 `trace.jsonl` —— **一行一条判定**，不是一行一个任务

```json
{"run_id":"...","task_id":"...","step":3,"node":"pickTool","answer":"search","confidence":0.83,"correct":true,"batch":7,"latency_ms":331}
```

**`confidence` 和 `correct` 必须在同一行。** 分桶 / ECE / Brier / AUC / 分离度 / 风险–覆盖
**全部**从这一对算出来 —— 少了任何一个，RQ2 就是空的。
`batch` 也要有：一次请求判多路时，**按批计时**，按记录求和会多算几倍（我们踩过，
`decisionMs` 一度比整轮墙钟还大）。

### 3.4 `usage.json` —— 逐调用，不是一个总数

```
每次 LLM 调用:  input_tokens_cached / input_tokens_uncached /
                output_tokens_visible / output_tokens_reasoning /
                usd / handshake_ms / ttft_ms / after_ttft_ms
每次判定请求:    questions_in_batch / handshake_ms / compute_ms
每次工具调用:    tool_name / ms / necessary | exploratory
整轮:            wall_ms / framework_ms / retry_ms / round_trips
```

★ **`framework_ms` 和 `retry_ms` 单独留。** 不留这两行，
就永远分不清差距里有多少是自己的开销、多少是对面在退避 ——
我们踩过：`generate` 那步墙钟 21.8s，meter 只记 11.0s，差掉的一半被误读成「慢 10 倍」。

### 3.5 `exit.json`

退出码 · 是否跑完 · 中断在第几任务第几步 · **中断原因原文**（不是「出错了」）。
**中断也要写**：`ran 137/300, stopped at task hotpotqa_0137, reason: provider 429 after 3 retries`。

### 3.6 ★ log 不在 git 里，所以要有归档位

`experiments/log/*` 在 `.gitignore` 里（体积）。**但「不进 git」不等于「可以不存」：**

| | |
|---|---|
| **机器上** | 跑完不许删。**跑挂的也不许删** |
| **归档** | ★ **跑完打包一份放共享盘**：`tar -czf <run_id>.tgz <run_dir>` + 记 checksum。**路径待定，定了写在这里** |
| **`result/` 里的每一行** | 带 `run_id` —— 这样就算日志归档在别处，也永远查得到是哪一次 |

**没归档的 log 只活在这台机器上**，而这台机器会重装。

---

## 4. 不许做的事

| | 为什么 |
|---|---|
| **手改 `result/` 里的数字** | 只许由脚本从 `log/` 生成。手改过的表**没有任何东西能验证它** |
| **删掉跑挂的那次 log** | 失败率本身是指标。而且「我们只留了成功的那几次」就是选择性报告 |
| **为了数字好看重跑** | 重跑可以，但**两次都要留**，并在表里注明跑了几次、为什么重跑 |
| **把 log / 数据写到 `/tmp`** | 别人找不到 = **没跑过**。而且重启就没了 |
| **写到 `experiments/` 以外** | 会被 `npm run check` 数进仓库（`dataset/` 和 `log/` 除外，它们在 ignore 里）|
| **一次运行充两个 arm** | arm 之间的差异是结论本身，不能共用一次运行 |
| **改了 prompt / 帧 / 门限之后不记版本** | 那两个 hash 就是为这件事留的 |
| **在脏工作区上跑正式数字** | `meta.json` 的 `dirty` 会记下来，然后这一行作废 |

---

## 5. 一次运行的完整流程

```
①  在伞仓库 docs/STATUS.md 声明持有（单写者，见 AGENTS.md §3）
②  git status 干净；记下 commit
③  ./scripts/run.ts 落 log/<dataset>/<arm>/<UTC>-seed<N>/    ← 目录先建，再跑
④  跑完检查 exit.json：跑完了吗？跑完了多少？
⑤  log/ 打包归档，记 checksum
⑥  从 log/ 生成汇总，写进 result/（每行带 run_id）
⑦  cd JevLoop && npx tsc --noEmit && npm run check && npm test && npm run demo
⑧  提交（显式列路径 + 路径检查 + 数量检查），清 STATUS
```

**★ ③ 的目录要「先建后跑」** —— 这样进程被 kill 的时候，你至少知道它跑到哪了。
**★ 报告只写实际跑过的命令和结果。** 没跑就说没跑，不许写「应该没问题」。

---

## 6. `result/` 是**单向**生成的

```
log/  ──(scripts/ 里的汇总脚本)──▶  result/
```

**不许反过来**（不许手填、不许从别处粘）。`result/` 里每个表旁边写一句：
**生成它的那条命令。** 没有那一句的表，等于没有 log。

**图也一样** —— 出图脚本进 `scripts/`，图进 `result/`，脚本里不许手写数据点。

---

## 7. 密钥

- `.env` **永不入库**（三个仓库的 `.gitignore` 都覆盖了）
- `cmd.txt` / `meta.json` / `stderr.txt` 里的 key **一律打掉**，
  写成 `sk-***`。stderr 尤其容易漏 —— provider 的错误信息经常把 key 一起打出来
- **写文件之前先确认 ignore 生效**（`git check-ignore -v <路径>`），不是之后
- ★ 跑之前先试一次：`git status --porcelain --untracked-files=all`
  —— **porcelain 有输出时也返回 0**，所以 `&& echo 干净` 这种写法永远打印「干净」。
  我们因为这句话在一个有别人未提交文件的仓库上提交过

---

## 8. 中断和失败

| 情况 | 怎么做 |
|---|---|
| Provider 429 / 超时 | **不要手动重试同一次运行** —— 那是两次运行。让它记进 `retry_ms`，或开新目录 |
| 跑到一半被杀 | `exit.json` 写清停在哪。**这一份照样归档、照样能进表**（标 `partial`）|
| 结果明显不对 | **先怀疑帧，不要先怀疑模型。** 帧里没有的，它判不出来（§8.2）—— 这是最常发生的一种 |
| 想改 prompt 再跑 | 新目录。`prompt_hash` 会不一样，这是好事 |

**报告失败时写清三样**：跑到第几个任务、停在第几步、原因原文。
「跑挂了」这三个字对别人零信息。
