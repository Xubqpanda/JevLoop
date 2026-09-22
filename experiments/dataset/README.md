# `dataset/` —— 数据在本地,**不进 git**

> ## ★ 一条分工,别混
>
> | | 装什么 | 进 git？ |
> |---|---|---|
> | **`benchmark/`** | **代码**：怎么加载、怎么判分 | ✅ |
> | **`dataset/`** | **数据本身**（下载到本机）+ **怎么把它拿回来** | ❌ 数据 / ✅ `DOWNLOADS.md` |
>
> **数据一个字节都不进 GitHub,但「怎么下载」必须写清楚** ——
> 否则别人 clone 下来只有一个空目录,这个仓库就不可复现了。

---

## 数据放哪

```
dataset/
├── README.md          这份（约定）
├── DOWNLOADS.md       ★ 生成的索引：每个数据集从哪来、拿哪个版本
└── hf/                HuggingFace 缓存（HF_HOME 指到这里）
    ├── hub/           按仓库组织
    └── datasets/      arrow 格式的产物
```

**全部在 `.gitignore` 里。**

---

## ★ `DOWNLOADS.md` 是**生成的**,不要手改

「怎么下载」这件事**住在代码里**,不在文档里:

```
Benchmark.downloads()   ──►   scripts/datasets.py --write   ──►   dataset/DOWNLOADS.md
   （权威，住在 loader 旁边）          （生成）                        （产物）
```

**为什么这样切**:文档里写一遍、loader 里再写一遍,两份就会分叉,
而分叉的后果是**别人照文档下载,拿到的和我们对不上** —— 那种错在跑完实验之后才会发现。

```sh
cd JevLoop
python3 -m experiments.scripts.datasets --list          # 看所有 loader 要什么
python3 -m experiments.scripts.datasets --check         # 哪些还没就位
python3 -m experiments.scripts.datasets --write         # 刷新本目录的 DOWNLOADS.md
python3 -m experiments.scripts.datasets --fetch all     # 下载
python3 -m experiments.scripts.datasets --fetch bfcl-simple
```

索引**只列有 loader 的数据集** —— 没有 loader 的东西不进表,否则它会变成愿望清单。

---

## 四种来源

| `kind` | 意思 | 自动下载？ |
|---|---|---|
| `hf-dataset` | HuggingFace 上的一个 `datasets` 数据集 | ✅ |
| `hf-file` | HF 仓库里的**单个文件**（BFCL 是 JSONL 散文件）| ✅ |
| `http` | 一个直链 | 手动 |
| **`build`** | **要现场建的东西**（Wikipedia 索引）| ❌ **故意不自动跑** |

`build` 不自动跑是**有意的**:它动辄几小时上百 GB,而且**版本没定就不能开工**。
让它需要人确认,比让它悄悄跑起来强。

---

## 给新 loader 加数据

在 loader 里实现 `downloads()`,返回一两条 `DownloadSpec`:

```python
def downloads(self) -> Sequence[DownloadSpec]:
    return [DownloadSpec(
        dataset=self.name,              # ★ 必须是这个 loader 的 name
        kind="hf-file",
        locator="org/repo",
        files=("data.json",),
        revision="main",                # ★ main = 没钉
        size_hint="~1 MB",
        note="JSONL 不是 JSON 数组",
    )]
```

然后 `--write` 刷新索引,`--fetch <name>` 下载。**三处都不用改文档。**

---

## 两道防线（都被绕过过一次,所以是两道）

1. **按目录**:`experiments/dataset/*` 在 `.gitignore` 里（配 `.gitkeep` 的否定）
2. **★ 按扩展名**:`*.arrow` / `*.parquet` / `*.safetensors` / `*.bin` … **任何路径下**都不进仓库

**为什么需要第二道**:第一道只挡一个路径。实测发生过 ——
`gsm8k` loader 的路径推导少了一层 `parent`,缓存落到了 `experiments/benchmark/dataset/`,
**13 个数据文件进了暂存区**,而路径检查（盯着 `experiments/dataset/`）完全没报警。
按扩展名挡是**与路径无关**的那一道。

同一类错误犯过两次（第二次把 `dataset/` 指到了 `JevLoop/dataset`),所以
`tests/test_datasets_index.py` 里有一条测试**把路径钉住**,改错了会红。
