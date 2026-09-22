"""`direct` —— 最简的一臂:**一次调用,不用工具,直接答**。

这是论文里的下界,也是**筛选阶段的唯一一臂**(`docs/PLAN-*.md` §0.9):
它便宜（每样本一次调用）、不依赖任何工具、不依赖我们的代码,
所以「这个 benchmark 还剩多少空间」这个问题可以先于一切基础设施被回答。

★ baseline 要对齐原论文并在代码里给引用 —— 这条从 `direct` 就开始守:
`direct` 就是标准 zero-shot prompting。定义出处:ReWOO, arXiv:2305.18323,
§3.1 Baselines ——「a standard zero-shot paradigm that prompts an LLM to
directly solve tasks or answer questions」。

★ 每个 baseline 都要这样标出处,**否则「含义一致」无从核对**。
"""

from __future__ import annotations

from dataclasses import dataclass

from experiments.core.agent import AgentOutcome, Session, action_answer
from experiments.core.models import Message
from experiments.core.types import Step


@dataclass
class Direct:
    """一次调用,一个答案。**没有任何循环。**

    `with_evidence=True` 是筛选阶段那一臂:`direct + 金标证据`。
    它给的是**上界估计** —— 如果连金标证据递到手里都答不出,
    那这个数据集是**难在生成**,我们的控制流层帮不上忙,不该为它写评估逻辑
    （`docs/PLAN-*.md` §0.9 的「`direct` 是单边筛子」那一条）。
    """

    with_evidence: bool = False

    @property
    def name(self) -> str:
        return "direct-oracle" if self.with_evidence else "direct"

    def solve(self, session: Session) -> AgentOutcome:
        parts = [session.task.prompt]
        if self.with_evidence:
            if not session.task.oracle_context:
                # ★ 不静默降级 —— 缺金标证据就等于这一臂没跑,不许悄悄当成普通 direct
                return AgentOutcome(error="direct-oracle: 这道题没有 oracle_context")
            parts.append(session.task.oracle_context)

        # ★ **一个字都不加。** 任务陈述和输出契约归 benchmark（见 core/bench.py 的说明）。
        #   早先这里追加过一句「Answer with the final answer only」,而 GSM8K 的题目里
        #   写着「Show your reasoning」—— 两条指令打架,模型听最后一条,
        #   于是这一臂实际测的是**提示词工程**,不是「不用工具直接答」。
        reply = session.call_model([Message(role="user", content="\n\n".join(parts))])
        answer = reply.text.strip()

        return AgentOutcome(
            steps=[Step(index=0, action=action_answer(answer))],
            final_answer=answer,
        )

# ★ 这里**故意什么都没有**。输出契约（「答案长什么样」）归 benchmark，
#   因为它是任务的一部分；baseline 只管**怎么和环境交互**。
#   各臂自己写一句输出格式 = 把提示词工程混进方法比较。
