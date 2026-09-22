"""Terminal-Bench —— **每题一个 Docker 容器环境，官方 pytest 判分。**

## 出处

| | |
|---|---|
| **论文** | *Terminal-Bench: A Benchmark for End-to-End Tasks in Real Terminal Environments*. arXiv:2601.11868 |
| **仓库** | `github.com/laude-institute/terminal-bench`（HEAD 没固定;改用 registry 给的 lock 版本）|
| **dataset** | **`terminal-bench-core==0.1.1`**（registry.json publish 的 latest release entry）<br>commit `91e10457b5410f16c44364da1a34cb6de8c488a5`<br>branch `dataset/terminal-bench-core/v0.1.x`<br>`task_id_subset` 80 个 id(含 `.easy`/`.hard` 变体) → 去点 **70 个 unique task dir**<br>每 task: `task.yaml` + `Dockerfile` + `docker-compose.yaml` + `run-tests.sh` + `tests/` |

## ★ 这是个**交互式容器环境**，形状和 ALFWorld 那类一样

每题是一个 docker 容器（`command: sleep infinity`），agent 通过 `docker exec` 在容器里跑命令来「做题」，
判分是**在 agent 改过的容器终态上跑官方 `run-tests.sh`**（它跑 `uv run pytest tests/`）。
「对不对」是 **pytest 的通过/失败** —— 这是官方 judge，不自己写判分器。

## ★ 判分口径（照学长群里第 ② 条：调官方别自己写）

`parser_name: pytest` 的 task 判据 = **`pytest` 的 pass/fail**。我们不重写 parser：
跑 `docker exec <container> sh -c 'bash ./run-tests.sh'`，读 pytest 的退出码与 LS 输出里的
`failed` 计数。correctness = `exit==0 且无 failed`。该任务的官方判据就是 pytest，所以这一步
不是「自己抄 parser」，是「跑官方给的 pytest 并读它的结论」。

## ⚠️ 必须记住的坑（读原仓库 + CLAUDE.md 读出来）

- **镜像每题现场 build**（`Dockerfile` 在 task 目录里）。100 题里大部分 task 各自一个镜像，
  这是跑这一项的主要时间成本（不是 deepseek 的钱，是 docker build）。
- **arm64 mac 上部分 task 跑不了**（如 pin 了 win32/x86 的 `3d-model-format-legacy`）。
  `docker build` 失败的那题，判分里如实记 `env_unavailable`，不静默当对。
- **不依赖 `terminal-bench` pip 包**（它要 Python 3.13 + postgres/supabase，太重，且它带自己的
  agent，我们要用自己的 `react` 臂）。这里直接用 `docker` CLI 管 build/exec/run-tests。

## ★ 抽样用 `rewoo_draw`（群里第 ③ 条点名要的）

  --seed 决定我们从 79 题里抽哪一批，全项目同一种做法。
"""

from __future__ import annotations

import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import yaml

from experiments.core.download import DATASET_DIR, DownloadSpec
from experiments.core.types import Judgment, Task, Tool, Trajectory

#: Terminal-Bench **published dataset** `terminal-bench-core==0.1.1`
#:   —— 不跟 main（main 一夜会换 task 集，不可复现）。registry.json 里登记的
#:   最 release 的 lock 版本: commit + branch + task_id_subset 三件全。
TB_COMMIT = "91e10457b5410f16c44364da1a34cb6de8c488a5"
TB_VERSION = "0.1.1"
TB_BRANCH = "dataset/terminal-bench-core/v0.1.x"

#: `v0.1.1` 分支下 task 定义在 `tasks/<task>/`（main 后来才改名 `original-tasks/`）。
TB_ROOT = DATASET_DIR / "terminal_bench" / "v0.1.1"
TASKS_DIR = TB_ROOT / "tasks"
#: v0.1.1 里 `datasets/terminal-bench-core-v0.yaml` 的 `task_ids`（80 个，含
#: `.base`/`.easy`/`.hard` 变体）**去后缀**的 70 个 unique task dir ——
#: 圈定的 published subset。不跑分支全集 86 个，只跑这 70。
#: （变体 .easy/.hard 是同一 task dir 不同难度，paper 一同报；我们这版每道跑一次，
#:  70 是 published subset 的 unique base，可复现。）
SUBSET_BASES: tuple[str, ...] = (
    'blind-maze-explorer-5x5', 'blind-maze-explorer-algorithm', 'build-initramfs-qemu', 'build-linux-kernel-qemu',
    'build-tcc-qemu', 'cartpole-rl-training', 'chess-best-move', 'conda-env-conflict-resolution',
    'configure-git-webserver', 'count-dataset-tokens', 'crack-7z-hash', 'create-bucket',
    'cron-broken-network', 'csv-to-parquet', 'decommissioning-service-with-sensitive-data', 'download-youtube',
    'eval-mteb', 'extract-moves-from-video', 'extract-safely', 'fibonacci-server',
    'fix-git', 'fix-pandas-version', 'fix-permissions', 'get-bitcoin-nodes',
    'git-multibranch', 'git-workflow-hack', 'gpt2-codegolf', 'grid-pattern-transform',
    'hello-world', 'heterogeneous-dates', 'hf-model-inference', 'incompatible-python-fasttext',
    'intrusion-detection', 'jupyter-notebook-server', 'modernize-fortran-build', 'new-encrypt-command',
    'nginx-request-logging', 'oom', 'openssl-selfsigned-cert', 'organization-json-generator',
    'password-recovery', 'path-tracing', 'path-tracing-reverse', 'play-zork',
    'polyglot-c-py', 'polyglot-rust-c', 'processing-pipeline', 'prove-plus-comm',
    'pytorch-model-cli', 'qemu-alpine-ssh', 'qemu-startup', 'raman-fitting',
    'reshard-c4-data', 'run-pdp11-code', 'sanitize-git-repo', 'security-vulhub-minio',
    'simple-sheets-put', 'simple-web-scraper', 'solana-data', 'sqlite-db-truncate',
    'sqlite-with-gcov', 'super-benchmark-upet', 'swe-bench-astropy-1', 'swe-bench-astropy-2',
    'swe-bench-fsspec', 'swe-bench-langcodes', 'tmux-advanced-workflow', 'train-fasttext',
    'vim-terminal-task', 'write-compressor',
)
IMAGE_PREFIX = "jevloop-tb"

EXEC_TOOL = Tool(
    name="exec",
    description=("Run a shell command in the task's container. The command runs in the task's "
                 "working directory; you see its stdout/stderr. Use this to inspect and modify "
                 "the environment to complete the task."),
    parameters={"type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"]},
)


class NeedsDocker(RuntimeError):
    """要用容器（exec/score）但 docker 不可用，或这一题的镜像在本机 build 不起来。"""


def _run(cmd: list[str], *, timeout: int = 1800, **kw) -> subprocess.CompletedProcess:
    """跑一条命令，超时和失败都抛清楚（不把 stderr 当 stdout 吞）。"""
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kw)


@dataclass
class TerminalBench:
    """Terminal-Bench v0.1.1 published dataset（70 unique task dirs）。`split` 只有 `test`。"""

    name: str = "terminal-bench"
    dataset_version: str = f"terminal-bench-core=={TB_VERSION}"
    data_dir: Path = field(default_factory=lambda: DATASET_DIR / "terminal_bench")
    split: str = "test"
    limit: int | None = None
    seed: int = 0
    #: agent 单题最多步数。TB task 有的要很久，runner 的 `--max-steps` 是默认 20，可经命令行覆盖。
    max_steps: int = 40
    #: docker build 单题超时（秒）。legacy build 慢，给宽一点。
    build_timeout_s: int = 1800
    #: run-tests.sh 超时（秒）。
    test_timeout_s: int = 600
    exec_timeout_s: int = 120
    #: 当前题。`score()` 在 agent 改过的**同一容器**终态上跑官方 `run-tests.sh`——
    #   不同于 ALFWorld 的「回放」，TB 的判据是容器状态被 agent 改成什么样了。
    _current: Task | None = field(default=None, init=False, repr=False)
    #: 这一题起出来的容器名。每题一个独立容器。
    _container: str | None = field(default=None, init=False, repr=False)

    # ── 数据 ────────────────────────────────────────────────

    def _task_dirs(self) -> list[Path]:
        if not TASKS_DIR.is_dir():
            raise FileNotFoundError(
                f"{self.name}: 缺 {TASKS_DIR}\n"
                f"  跑 `python3 -m experiments.scripts.datasets --fetch terminal-bench`"
            )
        all_dirs = sorted(d for d in TASKS_DIR.iterdir()
                          if d.is_dir() and (d / "task.yaml").exists())
        want = set(SUBSET_BASES)
        kept = [d for d in all_dirs if d.name in want]
        if len(kept) != len(SUBSET_BASES):
            have = {d.name for d in kept}
            missing = sorted(want - have)[:8]
            raise ValueError(
                f"{self.name}: subset 里有 {missing} 等 task 不在 {TASKS_DIR}。"
                f"  → 数据版本和 subset 不一致？"
            )
        return kept

    def tasks(self, *, split: str, limit: int | None, seed: int) -> Iterator[Task]:
        """出题。只取 0.1.1 `task_id_subset` 的 70 个 unique base dir;抽样 `rewoo_draw(seed)`。"""
        from experiments.benchmark.rewoo_port import rewoo_draw

        _ = split or self.split  # 官方只有 `test`；不分 train/test
        dirs = self._task_dirs()
        for i in rewoo_draw(len(dirs), limit if limit is not None else self.limit, seed):
            d = dirs[i]
            meta = yaml.safe_load((d / "task.yaml").read_text(encoding="utf-8")) or {}
            instruction = str(meta.get("instruction") or "").strip()
            if not instruction:
                continue
            yield Task(
                task_id=f"terminal-bench/{d.name}",
                prompt=(instruction
                        + "\n\nYou have a shell tool (`exec`). Inspect and modify the environment "
                        "to complete the task; stop calling tools when finished."),
                gold=None,  # TB 没有字符串金标——判据是 pytest 在容器终态的结果
                oracle_context=None,
                meta={
                    "task_name": d.name,
                    "difficulty": meta.get("difficulty"),
                    "category": meta.get("category"),
                    "parser": meta.get("parser_name"),
                    "estimated_duration_sec": meta.get("estimated_duration_sec"),
                    "task_dir": str(d),
                    "answer_kind": "environment",
                },
            )

    # ── 工具与环境 ────────────────────────────────────────────

    def on_task(self, task: Task) -> None:
        """runner 每题调一次。停下上一题的容器（每题独立容器），记这一题。"""
        self._stop()
        self._current = task
        self._container = None

    def _has_image(self, task_dir: Path) -> bool:
        r = _run(["docker", "image", "inspect", f"{IMAGE_PREFIX}:{task_dir.name}"], timeout=10)
        return r.returncode == 0

    def _ensure_container(self) -> str:
        """懒建这一题的镜像 + 容器。第一次 `tool_impls()` / `score()` 调时触发。"""
        if self._container is not None:
            return self._container
        if self._current is None:
            raise NeedsDocker(f"{self.name}: 没绑任务，先调 on_task(task)")
        if not shutil.which("docker"):
            raise NeedsDocker(f"{self.name}: 需要 `docker`（build/exec/run-tests）——没装")

        task_dir = Path(self._current.meta["task_dir"])
        image = f"{IMAGE_PREFIX}:{task_dir.name}"

        if not self._has_image(task_dir):
            br = _run(["docker", "build", "-t", image, "-f", "Dockerfile", "."],
                      cwd=str(task_dir), timeout=self.build_timeout_s)
            if br.returncode != 0:
                # ★ 多半是这题在本机 build 不动（arm64 起不来 x86/win32 镜像）。
                #   不是「task 不可解」，是「环境跑不了这题」——分开记。
                tail = br.stderr[-800:] or br.stdout[-800:]
                raise NeedsDocker(
                    f"{self.name}: docker build 失败 ({task_dir.name})\n{tail}"
                )
        cont = f"jevloop-tb-{uuid.uuid4().hex[:10]}"
        # 把 task dir 的 `run-tests.sh` + `tests/` 挂进容器 `/app`（base image 默认 WORKDIR）。
        # ★ 大部分 Dockerfile **不 COPY 这两样**（broken-python 等只 `FROM base + RUN ...`），
        #   Terminal-Bench 官方 harness 是靠绑定挂载把它们弄进容器的——我们手动 run 时
        #   不挂就缺，判分找不到 `run-tests.sh`（实测）。`:ro` 让判分脚本和测试不被 agent 改动，
        #   但镜像 /app 里的业务文件 agent 仍可写。
        td = str(task_dir.resolve())
        rr = _run(["docker", "run", "-d", "--name", cont, "--workdir", "/app",
                   "-v", f"{td}/run-tests.sh:/app/run-tests.sh:ro",
                   "-v", f"{td}/tests:/app/tests:ro",
                   "-e", "TEST_DIR=./tests", image, "sleep", "infinity"],
                  timeout=60)
        if rr.returncode != 0:
            raise NeedsDocker(f"{self.name}: docker run 失败\n{rr.stderr[-800:]}")
        self._container = cont
        return cont

    def tools(self) -> Sequence[Tool]:
        self._ensure_container()
        return (EXEC_TOOL,)

    def tool_impls(self) -> dict[str, Any]:
        self._ensure_container()

        def exec_(command: str = "", **kwargs: Any) -> str:
            cmd = str(command or kwargs.get("arg") or "").strip()
            if not cmd:
                return "Error: empty command."
            r = _run(["docker", "exec", self._container, "sh", "-c", cmd],
                     timeout=self.exec_timeout_s)
            out = r.stdout
            if r.stderr:
                out += (("\n[stderr]\n" + r.stderr) if r.stderr.strip() else "")
            if r.returncode != 0:
                out += f"\n[exit={r.returncode}]"
            # 截一下，免得一条 ls -R 把 context 撑爆
            return out[-8000:] if len(out) > 8000 else out

        return {"exec": exec_}

    # ── 判分 ────────────────────────────────────────────────

    def score(self, task: Task, trajectory: Trajectory) -> Judgment:
        """在 agent 改过的同一容器里跑官方 `run-tests.sh`，读 pytest 的结果。

        correctness = run-tests.sh 退出 0 **且** 输出里没有 `failed` 计数 > 0。
        pytest 自身就是这一题的官方判据，所以这里**不重写 parser**——
        跑它给的脚本、读它给的结论。
        """
        # 即使 agent 一条 exec 没调也得能判分（必然失败，但要走完记账）
        try:
            self._ensure_container()
        except NeedsDocker as exc:
            return Judgment(correct=False, score=0.0, detail=str(exc),
                             failure_class="env_unavailable")

        task_dir = Path(task.meta["task_dir"])
        r = _run(["docker", "exec", self._container, "sh", "-c", "bash ./run-tests.sh"],
                 timeout=self.test_timeout_s)
        out = (r.stdout or "") + (("\n[stderr]\n" + r.stderr) if (r.stderr or "").strip() else "")
        failed = _count_failed(out)
        passed = (r.returncode == 0) and failed == 0

        detail = f"run-tests.sh exit={r.returncode}; pytest failed={failed}"
        if not r.stdout and r.stderr:
            detail = f"只输出到 stderr（exit={r.returncode}）：{r.stderr[-300:]}"
        return Judgment(
            correct=passed, score=1.0 if passed else 0.0,
            detail=detail,
            failure_class=None if passed else "task_not_completed",
        )

    def check(self, task: Task, answer: str) -> bool:
        """Reflexion 的 Evaluator 要的成败信号 —— TB 的成败在容器状态，不在答案字符串。"""
        raise NeedsDocker(
            f"{self.name}: 成败由容器里的官方 pytest 判定，不是对答案字符串打分。"
            f"用 score()（在容器跑 run-tests.sh），或让臂通过 session.check_answer 接上。"
        )

    # ── 下载 ────────────────────────────────────────────────

    def downloads(self) -> Sequence[DownloadSpec]:
        return [
            DownloadSpec(
                dataset=self.name,
                kind="http",
                # ★ registry.json 里登记的 v0.1.1: commit + branch 都写明
                locator=(f"https://codeload.github.com/laude-institute/terminal-bench"
                         f"/tar.gz/{TB_COMMIT}"),
                files=("tasks/", "docker/", "registry.json"),
                revision=TB_COMMIT,
                size_hint="~14 MB（repo @ v0.1.1，不含 docker 镜像）",
                note=(f"★ **terminal-bench-core=={TB_VERSION}** —— registry.json 里 publish "
                      f"的 lock entry:commit {TB_COMMIT[:8]} / branch {TB_BRANCH}。"
                      "解压到 `dataset/terminal_bench/v0.1.1/`（顶层名是 codeloud 的前缀）。"
                      "docker 镜像**不在**这份里,每题 Dockerfile 现场 build。"
                      "task_id_subset 80 个 id (含 .easy/.hard 变体) 的 unique base = 70 dir。"),
            ),
        ]

    # ── 清理 ────────────────────────────────────────────────

    def _stop(self) -> None:
        if self._container:
            _run(["docker", "stop", self._container], timeout=30)
            _run(["docker", "rm", "-f", self._container], timeout=30)
            self._container = None


_FAILED_RE = re.compile(r"(\d+)\s+failed", re.IGNORECASE)


def _count_failed(test_output: str) -> int:
    """从 pytest 输出里抠 `failed` 计数；抠不到返回 -1（未知，但不当作 0 —— 见 gsm8k 的口径教训）。"""
    m = _FAILED_RE.search(test_output)
    if not m:
        return 0 if test_output.strip() else -1
    return int(m.group(1))


__all__ = ["TerminalBench", "NeedsDocker", "TB_COMMIT", "TASKS_DIR", "EXEC_TOOL"]
