"""`react` —— Reasoning + Acting 交错。

**出处**:Yao, Zhao, Yu, Du, Shafran, Narasimhan, Cao.
*ReAct: Synergizing Reasoning and Acting in Language Models*, ICLR 2023.
arXiv:2210.03629。仓库 `ysymyth/ReAct`。

对齐的是原文的这几条（**改动任何一条都要在这里写明,否则这个臂就不是 ReAct**）:

| 原文 | 这里 |
|---|---|
| 每步输出 `Thought:` / `Action:` / `Observation:` 三段 | `common.run_loop` + `LoopConfig.with_thought=True` |
| **每步重发整个 scratchpad**（observation-dependent）| `build_prompt` 每次重建全文,**不是多轮 messages** |
| 工具**写在 prompt 里**,动作**靠文本解析** | `common.render_tools` + `common.parse_step` |
| 在 `\\nObservation` 处截断 | `LoopConfig.stop = REACT_STOP` —— 没有它模型会自己编观察 |
| `finish[answer]` 收尾 | `parse_step` 认 `finish` / `answer` |

★ **和 `act` 的唯一区别就是 `with_thought`。** 这不是注释里的差别,是配置里的差别 ——
我们现有的 `bench/react.ts` 里 `thought` 出现 **0** 次,所以那条臂跑出来的是 act,
而标签写着 ReAct。**两条臂必须是两个文件、两个名字,不许靠读 prompt 才发现。**

★ **一个顺带的收获**:推理模型的原生思考走 `reasoning_content`,而 ReAct 要求一个
**可见的** `Thought:`。两者重叠。我们照原文保留可见 Thought（否则不是 ReAct）,
同时把 `reasoning_content` 的 token 数如实记进 usage —— 于是
**「前沿模型上 ReAct prompt 有 vs 无 thought」这个消融是免费的**,
而它按 `POSITIONING.md` §5④ 的说法是已发表的空缺。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from experiments.baseline.common import LoopConfig, run_loop
from experiments.core.controller import Controller  # noqa: F401  (类型注解用)
from experiments.core.agent import AgentOutcome, Session

INSTRUCTION = (
    "Solve the task by interleaving reasoning and tool use. "
    "You may use the tools below as many times as needed, then give the final answer."
)

# 原文附录里 HotpotQA 的 few-shot 轨迹。⚠️ **我们现在是 zero-shot** ——
# 原文用 6 条 exemplar,而 ReWOO 为了可比照抄了同样的 6 条。
# 想和原文数字对齐就得补上,想和「零样本」这条线比就留空。**两种都行,但必须写明。**
DEFAULT_EXEMPLARS = ""


@dataclass
class ReAct:
    """ReAct 臂。**别在这里加记忆、加压缩、加重试策略** —— 那是别的臂的事。"""

    exemplars: str = DEFAULT_EXEMPLARS
    max_parse_retries: int = 2
    name: str = "react"
    # ★ **控制器** —— 论文第一根轴:同一个循环,换一个「谁来回答下一步」。
    #   `None` = `LLMController`（生成 + 解析,决定藏在生成里）。
    controller: "Controller | None" = None
    config: LoopConfig = field(init=False)

    def __post_init__(self) -> None:
        self.config = LoopConfig(
            name="react",
            instruction=INSTRUCTION,
            # ★ 这一行就是 ReAct 与 Act 的全部区别
            with_thought=True,
            exemplars=self.exemplars,
            max_parse_retries=self.max_parse_retries,
            # ★ **控制器** —— 论文第一根轴。默认 `None` = `LLMController`
            #   （生成 + 解析,决定藏在生成里,也就是今天在用的那个）。
            #   传 `TypedController` 就是同一循环的另一个格子。
            controller=self.controller,
        )

    def solve(self, session: Session) -> AgentOutcome:
        return run_loop(session, self.config)
