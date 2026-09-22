"""我们自己那条臂（JevLoop）。

★ **这里不是 `baseline/`。** 那个目录装六个要对比的范式,每个都要对着原始论文对齐。
这里是论文主张本身 —— 判定与生成分开的那条臂。

现在只有 `typed.py`。完整 JevLoop（七个节点 + 授权闸门 + 交付闸门）还没有循环,
所以它**暂时不是一条能整跑的臂** —— 这一点写在 `typed.py` 头部,不靠猜。
"""

from experiments.jloop.typed import *  # noqa: F401,F403
