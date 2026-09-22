"""数据目录、下载索引、以及**那个犯过两次的路径推导**。

跑法（在 `JevLoop/` 下）::

    python3 -m pytest experiments/tests -q

这个文件里最重要的一条是 `test_experiments_dir_points_at_experiments` ——
同一类路径错误在本项目里犯过**两次**,每次的后果都是数据落错地方。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]  # JevLoop/
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.benchmark.bfcl.bfcl import SUBSETS, VERSION, Bfcl  # noqa: E402
from experiments.benchmark.gsm8k.gsm8k import Gsm8k  # noqa: E402
from experiments.core import download  # noqa: E402
from experiments.core.download import (  # noqa: E402
    DownloadSpec,
    fetch,
    hf_cache_dir,
    hf_repo_id,
    is_present,
    render_index,
)


# ═══════════════════════════════════════════════════════════
# ★ 路径推导 —— 犯过两次的那一类
# ═══════════════════════════════════════════════════════════


def test_experiments_dir_points_at_experiments() -> None:
    """★★ **路径推导靠数 `.parent` 就是会数错,所以钉一条测试。**

    实测两次:

    1. `benchmark/gsm8k/gsm8k.py` 少了一层 → 缓存落到 `experiments/benchmark/dataset/`,
       **13 个数据文件进了暂存区**,而路径检查完全没报警（它盯的是另一个路径）。
    2. `core/download.py` 多了一层 → `dataset/` 指到了 `JevLoop/dataset`。

    两次都不是逻辑错,是数错了。所以不靠人记。
    """
    assert download.EXPERIMENTS_DIR.name == "experiments", (
        f"EXPERIMENTS_DIR 指到了 {download.EXPERIMENTS_DIR} —— 多半是 .parent 数错了"
    )
    assert download.EXPERIMENTS_DIR.parent.name == "JevLoop"
    assert download.DATASET_DIR == download.EXPERIMENTS_DIR / "dataset"


def test_every_loader_agrees_on_the_experiments_dir() -> None:
    """三个模块各算一遍 `EXPERIMENTS_DIR` —— **必须算出同一个值**。

    各算各的正是这个 bug 的来源:改了一处忘了另一处,而症状是「数据明明下载了却报缺失」。
    """
    from experiments.benchmark.bfcl import bfcl as bfcl_mod
    from experiments.benchmark.gsm8k import gsm8k as gsm8k_mod

    dirs = {
        str(download.EXPERIMENTS_DIR),
        str(bfcl_mod.EXPERIMENTS_DIR),
        str(gsm8k_mod.EXPERIMENTS_DIR),
    }
    assert len(dirs) == 1, f"各模块算出的 experiments/ 不一致: {sorted(dirs)}"


# ═══════════════════════════════════════════════════════════
# locator 解析 —— 「下载过了却报缺失」的另一个来源
# ═══════════════════════════════════════════════════════════


def test_hf_repo_id_strips_the_config_suffix() -> None:
    """★ `openai/gsm8k:main` 的仓库 id 是 `openai/gsm8k`。

    不切掉 `:main` 的话仓库名会变成 `gsm8k:main`,缓存目录名对不上,
    于是**明明下载过了却报缺失**（实测踩过）。
    """
    spec = DownloadSpec(dataset="x", kind="hf-dataset", locator="openai/gsm8k:main")
    assert hf_repo_id(spec) == "openai/gsm8k"
    assert hf_cache_dir(spec).name == "datasets--openai--gsm8k"


def test_hf_cache_dir_for_a_plain_repo() -> None:
    spec = DownloadSpec(dataset="x", kind="hf-file", locator="org/repo")
    assert hf_cache_dir(spec).name == "datasets--org--repo"


# ═══════════════════════════════════════════════════════════
# 声明本身
# ═══════════════════════════════════════════════════════════


def test_bfcl_loader_declares_which_version_it_is() -> None:
    assert VERSION == "v3", "改了版本要把这条测试和 loader 头部的说明一起改"


def test_gsm8k_declares_an_hf_dataset() -> None:
    specs = Gsm8k(rows={"test": []}).downloads()
    assert len(specs) == 1
    spec = specs[0]
    assert spec.dataset == "gsm8k" == Gsm8k(rows={"test": []}).name, "dataset 必须是 loader 的 name"
    assert spec.kind == "hf-dataset"
    assert spec.locator.startswith("openai/gsm8k")


def test_bfcl_declares_per_subset_because_each_is_its_own_loader() -> None:
    """★ 一个数据集两个子集 → **两条声明**,`dataset` 各写各自的 name。

    因为「谁需要这份数据」的那根线是 loader 的 `name`,不是 HF 仓库名。
    """
    rows = {"BFCL_v3_irrelevance.json": []}
    specs = Bfcl(subset="v3-irrelevance", rows=rows).downloads()
    assert len(specs) == 1
    assert specs[0].dataset == "bfcl-v3-irrelevance"
    assert specs[0].files == ("BFCL_v3_irrelevance.json",), "irrelevance 没有金标文件,只有题目"


def test_bfcl_subset_names_carry_the_version() -> None:
    """★★ **`bfcl-simple` 这个词在 v3 和 v4 下含义不同**,所以名字必须带版本。

    实测踩过:HF 上 `gorilla-llm/Berkeley-Function-Calling-Leaderboard` 里的文件
    **全叫 `BFCL_v3_*`** —— v4 的 agentic 部分（web search / memory / format
    sensitivity）只在 GitHub 仓库里。所以「我们跑了 BFCL」这句话如果不带版本,
    读的人会以为是 v4,而 v4 的子集构成和判分权重都不一样。
    """
    for subset in SUBSETS:
        assert subset.startswith("v3-"), f"子集名 {subset!r} 没带版本"
        assert Bfcl(subset=subset, rows={SUBSETS[subset][0]: []}).name == f"bfcl-{subset}"


def test_bfcl_simple_declares_both_question_and_answer_files() -> None:
    rows = {"BFCL_v3_simple.json": [], "possible_answer/BFCL_v3_simple.json": []}
    files = Bfcl(subset="v3-simple", rows=rows).downloads()[0].files
    assert "BFCL_v3_simple.json" in files
    assert "possible_answer/BFCL_v3_simple.json" in files, "simple 的金标在 possible_answer/ 下"


def test_unpinned_revision_is_advertised_as_unpinned() -> None:
    """★ `main` 是**没钉**,不是「钉了 main」。索引里必须显眼地说出来,
    否则读的人会以为版本是确定的。"""
    spec = DownloadSpec(dataset="x", kind="hf-dataset", locator="a/b")
    assert not spec.pinned
    assert "未钉" in render_index([spec])
    assert DownloadSpec(dataset="x", kind="hf-dataset", locator="a/b",
                        revision="deadbeef").pinned


# ═══════════════════════════════════════════════════════════
# 索引渲染
# ═══════════════════════════════════════════════════════════


def test_index_says_it_is_generated() -> None:
    text = render_index([])
    assert "生成的" in text and "不要手改" in text
    assert "datasets --write" in text


def test_index_handles_the_empty_case() -> None:
    """没有 loader 时说的话,比一张空表有用 —— 它说明**为什么**是空的。"""
    assert "还没有任何 loader" in render_index([])


def test_index_lists_every_spec() -> None:
    specs = [
        DownloadSpec(dataset="a", kind="hf-dataset", locator="o/a"),
        DownloadSpec(dataset="b", kind="hf-file", locator="o/b", files=("f", "g")),
    ]
    text = render_index(specs)
    assert "`a`" in text and "`b`" in text
    assert "`2` 个" in text, "hf-file 要报文件数"


# ═══════════════════════════════════════════════════════════
# build 那类**故意不自动跑**
# ═══════════════════════════════════════════════════════════


def test_build_sources_refuse_to_download_automatically() -> None:
    """★ 几小时上百 GB 的东西,而且版本没定就不能开工 —— 要人确认是对的。

    报错里必须带**配方**和**说明**,否则人也不知道该怎么办。
    """
    spec = DownloadSpec(dataset="wiki", kind="build", locator="scripts/build_index.sh",
                        note="先固定 dump 版本")
    assert is_present(spec) is False, "build 的产物形态这里不知道,不许假装知道"
    with pytest.raises(RuntimeError, match="现场建"):
        fetch(spec)
    with pytest.raises(RuntimeError):
        fetch(spec)
    try:
        fetch(spec)
    except RuntimeError as exc:
        assert "scripts/build_index.sh" in str(exc) and "先固定 dump 版本" in str(exc)


def test_unknown_kind_fails_loudly() -> None:
    with pytest.raises(RuntimeError, match="不能自动下载"):
        fetch(DownloadSpec(dataset="x", kind="ftp", locator="ftp://x"))
