"""`log/<dataset>/<arm>/<UTC>-seed<N>/` 的读写。

规范在 `docs/PROTOCOL-experiments-*.md` §1/§3。三条不能破的:

1. **一次运行一个目录,永不覆盖。** 重跑就是新目录。
2. **跑挂的也留。** 失败率本身是指标,而且「我们只留了成功的那几次」就是选择性报告。
3. **目录先建后跑。** 这样进程被 kill 的时候,至少知道它跑到哪了。

这个模块只做文件;不做判断、不碰网络。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = EXPERIMENTS_DIR / "log"


def utc_stamp() -> str:
    """目录名里的时间戳。**UTC,而且文件名安全**（冒号在 Windows 上不能用）。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def repo_commit(repo: Path) -> tuple[str, bool]:
    """(commit, 脏不脏)。拿不到就返回 ('unknown', True) —— **不假装干净**。

    ★ 脏工作区跑出来的数字别人复现不了,连跑它的人自己都复现不了。
    """
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        porcelain = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
        return commit, bool(porcelain.strip())
    except (subprocess.SubprocessError, OSError):
        return "unknown", True


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


class RunLog:
    """一次运行的目录。用 `with` 保证 `exit.json` **一定**被写出来。

    即使中途抛异常,`__exit__` 也会落下 `exit.json` —— 一份没有 exit.json 的
    运行,事后分不清「跑完了」和「被 kill 了」。
    """

    def __init__(self, dataset: str, arm: str, seed: int, *, root: Path | None = None) -> None:
        self.dir = (root or LOG_DIR) / dataset / arm / f"{utc_stamp()}-seed{seed}"
        self.dir.mkdir(parents=True, exist_ok=False)  # exist_ok=False:同一个戳撞车要炸
        self.run_id = str(self.dir.relative_to(root or LOG_DIR))
        self._exit_written = False
        self._finished = 0
        self._total = 0
        self._stop_reason: str | None = None

    # —— 写 ───────────────────────────────────────────────────

    def _write(self, name: str, text: str) -> None:
        (self.dir / name).write_text(text, encoding="utf-8")

    def write_cmd(self, argv: Iterable[str], env: dict[str, str] | None = None) -> None:
        """完整命令行 + 生效的环境变量。**密钥打掉。**"""
        redacted = {
            k: ("***" if any(s in k.upper() for s in ("KEY", "TOKEN", "SECRET", "PASSWORD")) else v)
            for k, v in (env or {}).items()
        }
        lines = [" ".join(argv), "", "# 环境（敏感值已打掉）"]
        lines += [f"{k}={v}" for k, v in sorted(redacted.items())]
        self._write("cmd.txt", "\n".join(lines) + "\n")

    def write_meta(self, meta: Any) -> None:
        self._write("meta.json", json.dumps(_jsonable(meta), ensure_ascii=False, indent=2, sort_keys=True))

    def append_jsonl(self, name: str, record: Any) -> None:
        with (self.dir / name).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(_jsonable(record), ensure_ascii=False, sort_keys=True) + "\n")

    def progress(self, done: int, total: int) -> None:
        self._finished, self._total = done, total

    def stop(self, reason: str) -> None:
        """记下**中断原因原文**。「跑挂了」这三个字对别人零信息。"""
        self._stop_reason = reason

    def finish(self, exit_code: int = 0) -> None:
        if self._exit_written:
            return
        self._write(
            "exit.json",
            json.dumps(
                {
                    "exit_code": exit_code,
                    "completed": self._total > 0 and self._finished >= self._total,
                    "finished_tasks": self._finished,
                    "total_tasks": self._total,
                    "stop_reason": self._stop_reason,
                },
                ensure_ascii=False, indent=2,
            ),
        )
        self._exit_written = True

    def __enter__(self) -> "RunLog":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is not None:
            self.stop(f"{exc_type.__name__}: {exc}")
            self.finish(exit_code=1)
        else:
            self.finish(exit_code=0)
        return False  # 不吞异常 —— 吞了就等于「不假装成功」那条被破了


def archive_hint(run_dir: Path) -> str:
    """PROTOCOL §3.6:log 不在 git 里,所以跑完要打包归档。

    **返回一句话而不是替人打包** —— 归档位还没定（PROTOCOL 里写着「路径待定」）,
    这里悄悄打个包放进仓库反而更糟。
    """
    return f"tar -czf {run_dir.name}.tgz -C {run_dir.parent} {run_dir.name}   # 然后放到共享盘"


def disk_usage_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


__all__ = ["RunLog", "utc_stamp", "repo_commit", "archive_hint", "disk_usage_mb", "LOG_DIR",
           "EXPERIMENTS_DIR", "shutil", "os", "sys", "EXPERIMENTS_DIR"]
