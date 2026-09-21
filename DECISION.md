# DECISION.md

这个 agent 在 loop 里要问哪些问题、每个问题归谁答。

`docs/CODE-STYLE.md` 讲的是代码怎么写，这份文件讲的是**判断怎么下**。
它是给两个消费者读的：结构块编译成判定（不花钱），`## generator`
那一段进 system prompt（花钱）。

所以下面每写一个 `kind: choice`，都是**少问大模型一次**。

判定 id 旁边的括号是 `src/decisions.ts` 里对应的实现。

---

## needs_tool

kind: noul
when: 每个 step 的开头。判否就直接跳到生成，整个工具循环省掉

ask: The agent still has work to do before it can answer the task — an action the task requires that has not been taken yet

- true — the task still requires an action that has not happened: reading something, listing something, writing something, running something
- false — every action the task asks for has already been taken, and there is enough information to answer

policy:
  - prob:needs_tool >= 0.5 → use_tool
  - else → answer

★ **判据是「任务还有没有没做的动作」，不是「还有没有没拿到的信息」。**
这两个在只读任务上恰好一致，而在**写任务**上分道扬镳 —— 实测
（2026-09-21）：任务「把 alpha.ts 里的 totalOf 抄到一个新文件 summary.ts
里」，读完 alpha.ts 之后这个节点判了 `answer`（0.36），于是 loop 直接去
生成回答，**文件从没被写出来**。

它当时看到的是：任务、`already_done: "list_dir, read_file"`、以及 alpha.ts
的完整内容。**信息确实齐了** —— 按旧措辞「no tool call now would mean
answering with information it does not have yet」，答案就是「不需要工具」，
它判得没错。**是措辞把「写」这件事排除在问题之外了。**

配套：帧里必须有 `already_done`（一份**清单**，不是一个计数）——
`steps_done: 2` 那种写法分不出「读过了」和「写过了」（§8.2）。

常规 agent 也「判断」这件事，但方式是让大模型输出一段话来表达它 ——
于是这个只需一次前向的是非题，付了生成的价格。

## pick_tool

kind: choice
when: 判定需要动手之后，每一步都问一次
dynamic: toolsFor(ctx) —— 候选每步重建，下面列的是默认全集

### tool

ask: Which tool should the agent call next?

- read_file — 需要文件内容才能继续，且这个文件还没读过
- list_dir — 还不知道目录里有什么
- write_file — 要写的内容已经拿到，且目标路径明确
- done — 已有足够证据回答任务，工具循环可以结束了

policy:
  - top >= 0.6 → call
  - else → escalate

**候选是每步重建的，上面这份只是默认全集。** 固定的候选列表会让模型去选
一个已经不适用的动作 —— 实测写完文件之后 `write_file` 还在候选里，模型会再选它。
真正发给模型的是 `toolsFor(ctx)`，它把做过的动作删掉。

同时 `top >= 0.6` 用的是**选中项的概率**，不是 `confidence`。Laya 的
`confidence` 是归一化香农熵（`p=[0.8,0.2]` → `0.269`），拿它卡阈值时
在 2 个选项和 20 个选项下含义完全不同。

## pick_input

kind: choice
when: 选定的工具需要参数时（`list_dir` 不需要，它没有可挑的东西）
dynamic: unreadFiles(ctx) —— 还没读过的文件，每步重建

### file

ask: Which file should this tool call target?

- 还没读过的文件 — 候选由 `unreadFiles(ctx)` 每步算出来，这里不列举

policy:
  - top >= 0.5 → use
  - else → escalate

这条以前**不存在**，工具参数是写死的代码（永远返回 `files[0]`）。
配合「做过的动作从候选里删掉」，结果是 `read_file` 在一个 agent 生命周期里
只能触发一次、且只能读第一个文件 —— 「读取目录里的**全部** TypeScript 文件」
这种任务在那个实现下不可能完成。

按三分法，「读哪个文件」是**挑选**，该问判定；「写什么内容」是**生成**，
仍由调用方提供。

## grade_risk

kind: mixed
when: 每次真正调用工具之前

### risk

ask: How risky is this tool call?

- read-only
- reversible write
- irreversible
- destructive

### needs_auth

ask: This call must be explicitly authorised by a human before it runs

- true — it can destroy data, spend money, or leave the machine
- false — it only reads or writes inside the working directory

policy:
  - score:risk >= 2 → ask_human
  - prob:needs_auth >= 0.5 → ask_human
  - score:risk >= 1 → auto_audit
  - else → auto

**这是一个判定问两件事的例子**，所以 kind 是 `mixed`。两条闸门是**独立的**：
第一条是硬规则（风险分够高就必须授权，不接受概率绕过），第二条才是模型判断。
实测有效 —— 把风险阈值从 2 提到 3 试图绕开第一条时，第二条 `needs_auth=0.95` 兜住了。

`auto_audit` 承诺了留痕就必须真的留痕。以前这条分支和 `auto` 完全一样，
只多打一行 trace。

## step_ok

kind: noul
when: 每次工具执行之后

### ok

ask: This tool call itself completed and returned output this step can use. Judge only this call; whether the whole task is finished is a different question, decided elsewhere.

- true — the tool ran and returned content — no error, no empty result, and the target it names is the one that was requested
- false — the call did not work: an error, an empty result, a missing file, or output that clearly did not come from this tool

policy:
  - prob:ok >= 0.6 → continue
  - else → stop

**它判的是这一步，不是这个任务。** 决策帧里**故意没有 `task`** —— 这是修出来的：
以前帧带着 `task`、问题写着 "for the task"、判据写着 "what the task needed"，
三处一起把它拉到了任务级。实测任务「读一下 invoice.ts」时，第一步 `list_dir`
返回文件列表，它确实**成功**了，但没回答「这个文件定义了哪些函数」，
于是 `ok=0.470` 判否 → 动作 `stop` → **整个循环结束**。
任何需要多于一个工具的任务都跑不完。

「任务完成了吗」是 `isDone` 的职责，两者判错了层就会互相打架。

**门限是 0.6 不是 0.5，这条有讲究。** `noul` 的 0.5 是**最不确定**的取值，
而上面那道门限是闭区间 `>=` —— 一个等于「毫无信息」的值不该能放行任何事。
`step_ok` 又是七个判定点里唯一一个「放行 = 当没事发生」的门：它放行的意思是
「这一步成功了，继续」，于是**失败被吞掉**。用 Mock 跑一遍就能看见：
`noul` 恒 0.5，门限 0.5 时它会判 `continue`，把「完全不确定」读成了「成功」。

动作名只承诺实际发生的事。它以前叫 `retry_or_stop`，但**重试需要一个错误
分类策略，而那个策略不存在** —— 所以「retry」不能写进动作名里。

## is_done

kind: noul
when: 每次工具成功之后

### done

ask: The agent has done everything the task requires; any further tool call would not add information or change the outcome

- true — the goal stated in the task has been reached
- false — something the task asks for is still missing

policy:
  - prob:done >= 0.6 → finish
  - else → keep_going

语义早停，不是 `max_iter` 硬切。简单任务能立刻结束，而不是傻等到迭代上限。

## can_deliver

kind: mixed
when: 生成之后，回答发出去之前

### deliverable

ask: The answer carries out what the task asked for, and reports it accurately

- true — the task's request has been carried out and the answer describes it consistently with what the tools returned
- false — the task's request has not been carried out, or the answer misreports what happened

### unsupported

ask: The answer states something that the tool output does not support

- true — it claims a fact, file or result that was never observed
- false — everything it says traces back to a tool result

policy:
  - prob:unsupported >= 0.5 → revise
  - prob:deliverable >= 0.6 → deliver
  - else → revise

★★ **「把任务要求的事做了、并如实报告」—— 这是交付闸门，不是质量评审。**

旧措辞是「complete and correct for the task, and can be returned as-is」。它
把**回答额外提出的顾虑**也算成了「不完整」。实测（2026-09-21，帧一动不动、
只换这一句）：

    任务「把 alpha.ts 里的 totalOf 抄到一个新文件 summary.ts 里」

    回答 A：已完成……注意：只写入了 totalOf，没有写入 Order 接口
            → 旧措辞 d=0.28~0.50 → revise   新措辞 d=0.77~0.92 → **deliver**
    回答 B：（修订版）不能只写 totalOf，应为：```import type { Order } …```
            → 旧措辞 revise              新措辞 **仍然 revise** ✓

**关键是它没有把该拒的一起放进来**：回答 B 提议的是**从未写入盘上的内容**，
`unsupported` 照样 0.59~0.88，照样拦下。旧措辞过 0/8，新措辞过 4/8，
而**多过的那四个正是该过的**。

为什么这件事要紧：任务只说「抄那个函数」，**没要求那个文件能独立编译**。
回答 A 如实报告了「它不能独立编译」——那是**有用的额外信息**，不是没完成。
旧措辞把它读成「有问题 → 不交付 → 再试一次」，而**再试只会更糟**：
修订版开始提议改写文件内容，而那正是闸门该拦的。

§8.10 的「不假装成功」在这里的另一面：**也不要把说真话的判成不合格。**

以前生成完直接返回，靠事后人工抽查。现在每条输出都过一遍闸门。

`unsupported` 排在 `deliverable` **前面**不是随手写的：一个完整但凭空编了
事实的回答，比一个不完整的回答更该被打回。

**决策帧的预算直接决定这条判定的准确性。** 交付闸门要拿工具结果逐句核对回答，
把证据 clip 到 100 字符时它会正确地判出「回答里有证据不支持的内容」——
判定是对的，是帧喂少了。实测改成 600 字符后立刻通过。

## generator

（这一段**原样进 system prompt**。它也住在文件里 —— 换掉它不需要改代码。）

You have already decided what to do and you have the evidence the tools
returned. Your job is to **write the answer**, not to decide anything again.

- Answer the task using only the evidence provided. State only what the
  evidence supports; do not invent files, functions or results.
- **Reply in the same language as the task.**
- Answer the task directly. Do not narrate the process and do not explain
  how you decided.
- If the evidence is not enough to answer, say what is missing. Do not guess.

你已经通过判定确定了下一步动作，也拿到了工具返回的证据。你的工作是
**生成回答**，不是重新决定做什么。上面那四条是硬要求：只用证据、
**用任务的语言回答**、直接回答不复述过程、证据不够就明说缺什么。
