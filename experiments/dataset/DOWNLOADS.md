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
| `bfcl-v3-irrelevance` | hf-file | `gorilla-llm/Berkeley-Function-Calling-Leaderboard` | `1` 个 | ⚠️ **main**（未钉） | ~1 MB（JSONL 散文件） | **JSONL 不是 JSON 数组**；只有题目文件，金标在 possible_answer/ 下 |
| `bfcl-v3-live-simple` | hf-file | `gorilla-llm/Berkeley-Function-Calling-Leaderboard` | `2` 个 | ⚠️ **main**（未钉） | ~1 MB（JSONL 散文件） | **JSONL 不是 JSON 数组**；只有题目文件，金标在 possible_answer/ 下 |
| `bfcl-v3-multiple` | hf-file | `gorilla-llm/Berkeley-Function-Calling-Leaderboard` | `2` 个 | ⚠️ **main**（未钉） | ~1 MB（JSONL 散文件） | **JSONL 不是 JSON 数组**；只有题目文件，金标在 possible_answer/ 下 |
| `bfcl-v3-simple` | hf-file | `gorilla-llm/Berkeley-Function-Calling-Leaderboard` | `2` 个 | ⚠️ **main**（未钉） | ~1 MB（JSONL 散文件） | **JSONL 不是 JSON 数组**；只有题目文件，金标在 possible_answer/ 下 |
| `gsm8k` | hf-dataset | `openai/gsm8k:main` | — | ⚠️ **main**（未钉） | ~2 MB（7,473 + 1,319 条） | 纯文本，无前置；`revision` 未钉 —— 已用 fingerprint 记进 dataset_version |

## 怎么用

```sh
cd JevLoop
python3 -m experiments.scripts.datasets --check     # 哪些还没下载
python3 -m experiments.scripts.datasets --fetch all # 全部下载
python3 -m experiments.scripts.datasets --fetch <数据集名>
```

`kind=build` 的那些**不会自动下载**，要人确认 —— 它们动辄几小时上百 GB。
