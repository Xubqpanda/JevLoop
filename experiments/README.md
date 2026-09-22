# experiments/

评测脚本落在这里。

| 目录 | 放什么 |
|---|---|
| `dataset/` | 数据集与检索索引。**上百 GB，不进 git** |
| `benchmark/` | 各数据集的加载与评分 |
| `baseline/` | 各 agent loop 范式的实现 |
| `scripts/` | 入口命令 |
| `log/` | 每次运行的原始产物。**不进 git** |
| `result/` | 汇总后的表和图 |

**为什么 `dataset/` 和 `log/` 不进 git**：一个检索索引加十几个数据集是上百 GB，
一次跑出来的逐条日志是几十 MB。**仓库不是硬盘。**

`.gitignore` 用的是 `dir/*` 加否定 `.gitkeep`，**不是**直接忽略整个目录 ——
后者连占位文件一起丢掉，于是别人 clone 出来这些目录**根本不存在**，而这份文件说它们存在。

**提交之前**：见 `CONTRIBUTING.md` 与 `docs/CODE-STYLE.md`。
