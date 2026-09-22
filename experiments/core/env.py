"""`.env` 加载器 —— 照抄 TS 的 `src/env.ts`。

## 为什么需要它

**实测**:Python 侧原来**根本没有加载器** —— 只读 `os.environ`。
而 key 和判定后端的 URL 都躺在伞仓库根的 `.env` 里。结果是:

```
python3 -m experiments.scripts.run --model deepseek-chat ...
  → "缺 API key：给 --api-key，或设 DEEPSEEK_API_KEY / OPENAI_API_KEY。"
```

报错信息本身是对的,但它把人指向「去设一个环境变量」,
而**真正该做的是让这个仓库读它自己那份 `.env`** —— 和 TS 侧一样。
**一份配置散在两处读法里,就是一个必然的坑。**

★ 注意这条的形状:它不是「少了个功能」,是**「东西在对的地方躺着而没人去拿」**。
第 10 轮那条失效链、今天的 `top()` 分歧,全是这一个形状。

## 只做四件事

`KEY=VALUE` · `export KEY=VALUE` · `#` 注释 · 两端成对的引号。
**不做变量插值、不做多行** —— 需要的自己去装 dotenv。

## ★ 两条从 TS 侧搬过来的规矩

**① `export KEY=VALUE` 必须剥掉前缀。** 不剥的话键名会变成 `export KEY`,
于是「键不存在」,外加一个凭空多出来的 `export KEY`。用户以为设上了,
实际没有,表现为「没有 key → 退回 Mock」—— **排查方向被完全带偏。**

**② 认不出来的行跳过**并且**报出来**（返回值里的 `skipped`）,
绝不 `os.environ[key] = value`。**一个既不跳过也不报错的写法,比明确不支持更糟**:
它产出的是一个看起来成功的错东西（§8.10）。

## 不打印值

`loaded` 返回的是**键名**,不是值 —— 密钥不进日志。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

#: 键名的合法形状。★ 不校验时 `export KEY=VALUE` 会被当成键名的一部分。
_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass
class LoadEnvResult:
    """`loaded` 只装**键名** —— 值一个字节都不返回,免得被顺手打进日志。"""

    loaded: list[str] = field(default_factory=list)
    #: 认不出来、因而**跳过**的行（原文）。空 = 整份文件都懂了。
    skipped: list[str] = field(default_factory=list)
    path: str = ""

    def __bool__(self) -> bool:
        return bool(self.loaded)


def find_env_file(start: str | Path, *, levels: int = 3) -> Path | None:
    """从 `start` 开始往上找 `.env`,最多 `levels` 层。

    ★ `JevLoop/` 是独立仓库、要能**单独 clone 就跑** —— 所以它自己那份 `.env`
      优先（第一层）。伞仓库根的那份在第二层,是**开发时**的便利,
      不是它依赖的东西。
    """
    directory = Path(start).resolve()
    for _ in range(levels):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
        parent = directory.parent
        if parent == directory:
            break
        directory = parent
    return None


def load_env(start: str | Path | None = None, *,
             keep_existing: bool = True) -> LoadEnvResult:
    """找到并加载 `.env`。找不到就静默返回空的（**没有文件不是错误**）。"""
    path = find_env_file(start or Path.cwd())
    if path is None:
        return LoadEnvResult()

    out = LoadEnvResult(path=str(path))
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        # `export KEY=VALUE` —— 剥掉前缀再解析,与 dotenv 行为一致
        if line.startswith("export "):
            line = line[len("export "):].strip()

        eq = line.find("=")
        if eq <= 0:
            out.skipped.append(raw.strip())
            continue

        key = line[:eq].strip()
        value = line[eq + 1:].strip()
        # 去掉两端成对的引号
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]

        # ★ 键名不合法的行**跳过并报出来**,绝不写进环境。
        #   以前 TS 侧这里没有这一道,于是 `export KEY` 设了一个叫 `export KEY`
        #   的变量**还把它列进返回值** —— 调用方打印「已加载 …」,看起来完全成功。
        if not _KEY_RE.match(key):
            out.skipped.append(raw.strip())
            continue

        if keep_existing and key in os.environ:
            continue
        os.environ[key] = value
        out.loaded.append(key)

    return out


__all__ = ["load_env", "find_env_file", "LoadEnvResult"]
