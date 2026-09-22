# 数据下载索引

> ## ⚠️ 这个文件是**生成的**，不要手改。
>
> ```sh
> python3 -m experiments.scripts.datasets --write
> ```
>
> 权威来源是**每个 loader 自己的 `downloads()`** —— 「怎么下载」住在代码旁边，
> 文档里再抄一遍就会分叉，而分叉的后果是**别人照文档下载，拿到的和我们对不上**。

**数据不进 git。** 下载到本地服务器，落在 `experiments/dataset/`。
这份文件只回答一个问题：**怎么把它拿回来，拿哪个版本。**

| 数据集（loader）| 来源 | 定位 | 文件 | 版本 | 大概多大 | 备注 |
|---|---|---|---|---|---|---|
| `alfworld` | http | `https://codeload.github.com/alfworld/alfworld/tar.gz/aaba6870f86c5be6a08a491f32a50b906227bc3e` | `2` 个 | aaba6870f86c5be6a08a491f32a50b906227bc3e | ~3.8 MB（整个仓库） | ★ `logic/` 下那两个文件（PDDL 领域 + TextWorld 语法）**不在数据包里,在仓库里** —— 少了它们环境起不来,而报错说的是「Unsupported game format」,不指向真正的原因。解压到 `dataset/alfworld/logic/` |
| `alfworld` | http | `https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_json.zip` | `1` 个 | aaba6870f86c5be6a08a491f32a50b906227bc3e | 69 MB | ★ 数据在 **release assets** 里,既不在仓库也不在 HF。★ **三个 zip 都要**:`solvable` 键在 `game.tw-pddl`（0.4.2 那个包）里,少了它过滤器会把全部题目筛掉。解压到 `dataset/alfworld/json_2.1.1/<split>/` |
| `alfworld` | http | `https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_pddl.zip` | `1` 个 | aaba6870f86c5be6a08a491f32a50b906227bc3e | 34 MB | ★ 数据在 **release assets** 里,既不在仓库也不在 HF。★ **三个 zip 都要**:`solvable` 键在 `game.tw-pddl`（0.4.2 那个包）里,少了它过滤器会把全部题目筛掉。解压到 `dataset/alfworld/json_2.1.1/<split>/` |
| `alfworld` | http | `https://github.com/alfworld/alfworld/releases/download/0.4.2/json_2.1.3_tw-pddl.zip` | `1` 个 | aaba6870f86c5be6a08a491f32a50b906227bc3e | 35 MB | ★ 数据在 **release assets** 里,既不在仓库也不在 HF。★ **三个 zip 都要**:`solvable` 键在 `game.tw-pddl`（0.4.2 那个包）里,少了它过滤器会把全部题目筛掉。解压到 `dataset/alfworld/json_2.1.1/<split>/` |
| `bfcl-v3-irrelevance` | hf-file | `gorilla-llm/Berkeley-Function-Calling-Leaderboard` | `1` 个 | ⚠️ **main**（未钉） | ~1 MB（JSONL 散文件） | **JSONL 不是 JSON 数组**；只有题目文件，金标在 possible_answer/ 下 |
| `bfcl-v3-live-simple` | hf-file | `gorilla-llm/Berkeley-Function-Calling-Leaderboard` | `2` 个 | ⚠️ **main**（未钉） | ~1 MB（JSONL 散文件） | **JSONL 不是 JSON 数组**；只有题目文件，金标在 possible_answer/ 下 |
| `bfcl-v3-multiple` | hf-file | `gorilla-llm/Berkeley-Function-Calling-Leaderboard` | `2` 个 | ⚠️ **main**（未钉） | ~1 MB（JSONL 散文件） | **JSONL 不是 JSON 数组**；只有题目文件，金标在 possible_answer/ 下 |
| `bfcl-v3-simple` | hf-file | `gorilla-llm/Berkeley-Function-Calling-Leaderboard` | `2` 个 | ⚠️ **main**（未钉） | ~1 MB（JSONL 散文件） | **JSONL 不是 JSON 数组**；只有题目文件，金标在 possible_answer/ 下 |
| `fever` | hf-dataset | `copenlu/fever_gold_evidence` | — | ⚠️ **main**（未钉） | ~50 MB（228,277 + 15,935 + 16,039） | ★ ReWOO 用的就是这个 HF 源,**不是**官方原始仓库。三个划分都带标签（实测）,没有 HotpotQA / TriviaQA 那个坑。`revision` 未钉,已用 fingerprint 记进 dataset_version |
| `gsm8k` | hf-dataset | `openai/gsm8k:main` | — | ⚠️ **main**（未钉） | ~2 MB（7,473 + 1,319 条） | 纯文本，无前置；`revision` 未钉 —— 已用 fingerprint 记进 dataset_version |
| `hotpotqa` | hf-dataset | `hotpot_qa:fullwiki` | — | ⚠️ **main**（未钉） | ~1 GB（90,447 + 7,405 × 2） | ★ config 必须是 `fullwiki` —— ReWOO 用的就是它,换成 `distractor` 就是另一个数据集。`revision` 未钉,已用 fingerprint 记进 dataset_version |
| `physicsquestions` | http | `https://codeload.github.com/billxbf/ReWOO/tar.gz/9cd0283043ff4be0c9d614fda2789d143ca6ffd1` | `1` 个 | 9cd0283043ff4be0c9d614fda2789d143ca6ffd1 | ~6 MB（整个 ReWOO 仓库） | ReWOO 仓库自带的 BigBench CSV。★ 一次下载抽三个文件。原始出处是 `google/BIG-bench` 的 `benchmark_tasks/<task>/task.json`，但**这里刻意用 ReWOO 那份** —— 我们和它比的就是这三个数，回原站会引入一个在数字上看不出来的版本差（而 `strategy_qa` 在 BigBench 主干上已被移除，实测 404） |
| `sotuqa` | http | `https://codeload.github.com/billxbf/ReWOO/tar.gz/9cd0283043ff4be0c9d614fda2789d143ca6ffd1` | `1` 个 | 9cd0283043ff4be0c9d614fda2789d143ca6ffd1 | ~6 MB（整个 ReWOO 仓库；本数据集 22 KB） | ★ ReWOO 自带的 curated 数据集（74 条），全项目最小的一个。★ 国情咨文全文 `data/docs/state_of_the_union.txt` 也在这个 tarball 里,要接 oracle 就把它加进 `files` |
| `sportsunderstanding` | http | `https://codeload.github.com/billxbf/ReWOO/tar.gz/9cd0283043ff4be0c9d614fda2789d143ca6ffd1` | `1` 个 | 9cd0283043ff4be0c9d614fda2789d143ca6ffd1 | ~6 MB（整个 ReWOO 仓库） | ReWOO 仓库自带的 BigBench CSV。★ 一次下载抽三个文件。原始出处是 `google/BIG-bench` 的 `benchmark_tasks/<task>/task.json`，但**这里刻意用 ReWOO 那份** —— 我们和它比的就是这三个数，回原站会引入一个在数字上看不出来的版本差（而 `strategy_qa` 在 BigBench 主干上已被移除，实测 404） |
| `strategyqa` | http | `https://codeload.github.com/billxbf/ReWOO/tar.gz/9cd0283043ff4be0c9d614fda2789d143ca6ffd1` | `1` 个 | 9cd0283043ff4be0c9d614fda2789d143ca6ffd1 | ~6 MB（整个 ReWOO 仓库） | ReWOO 仓库自带的 BigBench CSV。★ 一次下载抽三个文件。原始出处是 `google/BIG-bench` 的 `benchmark_tasks/<task>/task.json`，但**这里刻意用 ReWOO 那份** —— 我们和它比的就是这三个数，回原站会引入一个在数字上看不出来的版本差（而 `strategy_qa` 在 BigBench 主干上已被移除，实测 404） |
| `tau2-bench` | http | `https://codeload.github.com/sierra-research/tau2-bench/tar.gz/refs/tags/v0.1.0` | `9` 个 | 37199f36924c | ~56 MB（整个仓库；本数据集约 25 MB） | ★ **必须钉 v0.1.0** —— 仓库 HEAD 已经是 τ³-bench v1.0.1,而它自己的 README 写着 `<1.0.1` 的结果不能和 `>=1.0.1` 比。
  ★ 而 v0.1.0 的 `tasks.json` **就是 base 划分**（278 条）;HEAD 上多了 `split_tasks.json`,base 要从里面挑 —— 换版本连哪 278 条都会变。
  ★ `test` 那一列在排行榜口径下要跑 **pass^k = 4 次/任务**（`comb(success,k)/comb(trials,k)`） |
| `terminal-bench` | http | `https://codeload.github.com/laude-institute/terminal-bench/tar.gz/91e10457b5410f16c44364da1a34cb6de8c488a5` | `3` 个 | 91e10457b5410f16c44364da1a34cb6de8c488a5 | ~14 MB（repo @ v0.1.1，不含 docker 镜像） | ★ **terminal-bench-core==0.1.1** —— registry.json 里 publish 的 lock entry:commit 91e10457 / branch dataset/terminal-bench-core/v0.1.x。解压到 `dataset/terminal_bench/v0.1.1/`（顶层名是 codeloud 的前缀）。docker 镜像**不在**这份里,每题 Dockerfile 现场 build。task_id_subset 80 个 id (含 .easy/.hard 变体) 的 unique base = 70 dir。 |
| `triviaqa` | hf-dataset | `trivia_qa:rc.nocontext` | — | ⚠️ **main**（未钉） | ~700 MB（138,384 + 17,944 + 17,210） | ★ config 必须是 `rc.nocontext` —— ReWOO 用的就是它。★ 答案带**别名集**,官方口径与 ReWOO 口径不同,见本文件头部。`revision` 未钉,已用 fingerprint 记进 dataset_version |

## 怎么用

```sh
cd JevLoop
python3 -m experiments.scripts.datasets --check     # 哪些还没下载
python3 -m experiments.scripts.datasets --fetch all # 全部下载
python3 -m experiments.scripts.datasets --fetch <数据集名>
```

`kind=build` 的那些**不会自动下载**，要人确认 —— 它们动辄几小时上百 GB。
