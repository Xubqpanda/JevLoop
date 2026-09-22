"""各数据集的 loader / adapter。**一个数据集一个文件,文件名见表。**

实现 `core/bench.py` 的 `Benchmark`：`tasks()` / `tools()` / `tool_impls()` / `score()`。

★ `score()` 尽量调**官方评分器** —— 重写一遍官方判分逻辑 = 换判分器,
而换判分器 Acc 就不可比。
"""
