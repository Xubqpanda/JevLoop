"""数据从哪来。**「怎么下载」这件事住在代码里,不在文档里。**

## 为什么要这样切

`dataset/` 里放数据（下载到本地服务器,**不进 git**),`benchmark/` 里放代码。
但「每个数据集从哪来」这句话**必须有归属** —— 文档里写一遍、loader 里再写一遍,
两份就会分叉,而分叉的后果是**别人照文档下载,拿到的和我们对不上**。

所以规则是:

```
Benchmark.downloads()  ──►  scripts/datasets.py --write  ──►  dataset/DOWNLOADS.md
   （权威，住在 loader 旁边）        （生成）                      （不许手改）
```

**索引是生成的,不是手写的。** 它只列**有 loader 的数据集** ——
没有 loader 的东西不进这张表,因为它还不参与实验。

## 四种来源

| `kind` | 意思 | 怎么取 |
|---|---|---|
| `hf-dataset` | HuggingFace 上的一个 `datasets` 数据集 | `load_dataset(id, config)` |
| `hf-file` | HuggingFace 仓库里的**单个文件**（BFCL 是 JSONL 散文件）| `hf_hub_download` |
| `http` | 一个直链 | 手动或 `curl` |
| `build` | **要现场建的东西**（Wikipedia 索引就是）| 只记配方,不自动跑 |

★ `build` 这一类**故意不自动执行**:它动辄几小时上百 GB,而且**版本没定就不能开工**
（`docs/MANIFEST-datasets-*.md` 里那种 `待核`）。让它需要人确认,是对的。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

# 数据和本文档都在 `experiments/dataset/`,**不进 git**。
#
# ★ `parents[1]`,不是 `parents[2]`:这个文件在 `experiments/core/download.py`。
#   **同一类错误在本项目里犯过两次**（gsm8k 那次把缓存下到了
#   `experiments/benchmark/dataset/`,13 个数据文件一度进了暂存区;
#   这一次把 `dataset/` 指到了 `JevLoop/dataset`）。
#   路径推导靠数层级就是会数错 —— 所以 `tests/test_datasets_index.py`
#   里有一条测试把这个路径钉住,改错了会红。
EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = EXPERIMENTS_DIR / "dataset"
DEFAULT_HF_HOME = DATASET_DIR / "hf"
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
INDEX_NAME = "DOWNLOADS.md"


def use_local_data_dir() -> Path:
    """把 HF 的缓存指到 `experiments/dataset/hf/`。

    ★ 国内直连 huggingface.co 不通（实测 `HTTP 000`），hf-mirror.com 通。
    **已经设过 `HF_ENDPOINT` 就尊重它** —— 别替用户改主意。
    """
    os.environ.setdefault("HF_HOME", str(DEFAULT_HF_HOME))
    os.environ.setdefault("HF_ENDPOINT", DEFAULT_HF_ENDPOINT)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    return Path(os.environ["HF_HOME"])


@dataclass(frozen=True)
class DownloadSpec:
    """一个 loader 需要的一份数据。

    **`dataset` 必须是那个 loader 的 `name`** —— 它是把「这份数据」和
    「谁要用它」连起来的那根线。两条不同的题用同一份数据时,写两条。
    """

    dataset: str
    kind: str  # hf-dataset | hf-file | http | build
    locator: str
    files: tuple[str, ...] = ()
    # ★ 版本要钉。`main` 是**没钉**,不是「钉了 main」—— 它等于「哪天拉的算哪天」。
    revision: str = "main"
    size_hint: str = ""
    # 需要什么前置条件（比如「先固定 Wikipedia dump 版本」）
    note: str = ""

    @property
    def pinned(self) -> bool:
        return self.revision not in ("", "main", "unpinned")


def hf_repo_id(spec: DownloadSpec) -> str:
    """从 locator 里取出**仓库 id**。

    ★ `hf-dataset` 的 locator 形式是 `<repo>:<config>`（`openai/gsm8k:main`），
    **`:config` 必须先切掉**再分 org/name —— 否则仓库名会变成 `gsm8k:main`,
    而缓存目录名对不上,于是「明明下载过了却报缺失」（实测踩过）。
    """
    return spec.locator.partition(":")[0]


def hf_cache_dir(spec: DownloadSpec) -> Path:
    """`hf-dataset` / `hf-file` 在本地的落点（HF 的 hub 缓存按仓库组织）。"""
    org, _, name = hf_repo_id(spec).partition("/")
    return Path(os.environ.get("HF_HOME", str(DEFAULT_HF_HOME))) / "hub" / f"datasets--{org}--{name}"


def is_present(spec: DownloadSpec) -> bool:
    """这份数据在本地了吗。

    `build` 一律返回 False —— 它的产物形态由各自的配方决定,
    **这里不假装知道**（假装知道的后果是「检查通过但数据没建」）。
    """
    if spec.kind in ("hf-dataset", "hf-file"):
        return hf_cache_dir(spec).exists()
    return False


def fetch(spec: DownloadSpec) -> list[Path]:
    """真去下载。**只处理能自动处理的两种**,其余抛清楚的话。"""
    use_local_data_dir()

    if spec.kind == "hf-dataset":
        from datasets import load_dataset

        config, _, split = spec.locator.partition(":")
        load_dataset(config, split or None)
        return [hf_cache_dir(spec)]

    if spec.kind == "hf-file":
        from huggingface_hub import hf_hub_download

        if not spec.files:
            raise ValueError(f"{spec.dataset}: hf-file 必须列出 files")
        return [
            Path(hf_hub_download(spec.locator, f, repo_type="dataset", revision=spec.revision))
            for f in spec.files
        ]

    if spec.kind == "build":
        raise RuntimeError(
            f"{spec.dataset}: 这是要**现场建**的东西，不会自动跑。\n"
            f"  配方: {spec.locator}\n"
            f"  说明: {spec.note or '（没写）'}\n"
            f"  理由: 这一类动辄几小时上百 GB，而且版本没定就不能开工 —— 要人确认是对的。"
        )

    raise RuntimeError(f"{spec.dataset}: kind={spec.kind!r} 不能自动下载，请手动取 {spec.locator}")


def render_index(specs: Sequence[DownloadSpec]) -> str:
    """生成 `dataset/DOWNLOADS.md` 的内容。**别手改这个文件** —— 它是产物。"""
    lines = [
        "# 数据下载索引",
        "",
        "> ## ⚠️ 这个文件是**生成的**，不要手改。",
        ">",
        "> ```sh",
        "> python3 -m experiments.scripts.datasets --write",
        "> ```",
        ">",
        "> 权威来源是**每个 loader 自己的 `downloads()`** —— 「怎么下载」住在代码旁边，",
        "> 文档里再抄一遍就会分叉，而分叉的后果是**别人照文档下载，拿到的和我们对不上**。",
        "",
        "**数据不进 git。** 下载到本地服务器，落在 `experiments/dataset/`。",
        "这份文件只回答一个问题：**怎么把它拿回来，拿哪个版本。**",
        "",
    ]

    if not specs:
        lines += ["_还没有任何 loader 声明下载来源。_", ""]
        return "\n".join(lines)

    lines += [
        "| 数据集（loader）| 来源 | 定位 | 文件 | 版本 | 大概多大 | 备注 |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in sorted(specs, key=lambda x: (x.dataset, x.locator)):
        files = f"`{len(s.files)}` 个" if s.files else "—"
        # ★ 没钉版本要说出来，别让人误以为它钉了
        revision = s.revision if s.pinned else f"⚠️ **{s.revision}**（未钉）"
        lines.append(
            f"| `{s.dataset}` | {s.kind} | `{s.locator}` | {files} | {revision} | "
            f"{s.size_hint or '—'} | {s.note or '—'} |"
        )

    lines += [
        "",
        "## 怎么用",
        "",
        "```sh",
        "cd JevLoop",
        "python3 -m experiments.scripts.datasets --check     # 哪些还没下载",
        "python3 -m experiments.scripts.datasets --fetch all # 全部下载",
        "python3 -m experiments.scripts.datasets --fetch <数据集名>",
        "```",
        "",
        "`kind=build` 的那些**不会自动下载**，要人确认 —— 它们动辄几小时上百 GB。",
        "",
    ]
    return "\n".join(lines)
