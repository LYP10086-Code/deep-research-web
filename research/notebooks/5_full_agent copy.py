"""Full Multi-Agent Research System - 自包含可运行脚本（智谱 GLM 版）.

将 notebook 5 的端到端研究系统整理为独立脚本，整合前四个 notebook 的全部组件：
1. scope 阶段：scope_brief（判断请求/反馈是否清晰，产出或修订研究简报，单节点合并版）
2. supervisor 阶段：多智能体协调，并行派出 researcher 子代理迭代搜索
3. report 阶段：final_report_generation 汇总所有研究成果生成最终报告

改动说明（相对 src 源文件 / notebook 5）：
- 模型全部换为 openai:glm-5.3-flashx（.env 里的 ZHIPUAI_API_KEY，走智谱
  OpenAI 兼容端点），包括 notebook 5 原本使用 openai:gpt-4.1 的 writer_model
- 所有子图代码（scope / researcher / supervisor）内联到本文件，不依赖 src 中
  顶层实例化模型的模块，避免硬编码模型配置
- 激进合并：clarify_with_user 与 write_research_brief 合为单一 scope_brief 节点，
  统一用 BriefRevision 结构化输出决策——首跑请求模糊或修订反馈模糊时反问澄清，
  清晰时直接产出/修订简报；原 clarify 的提问规则（缩写澄清、不重复问等）已移植
- 所有澄清/审阅交互统一收敛到 review_brief 的 interrupt 人工入口，图内不再有假 END
- 用 astream(stream_mode="updates") 打印每个节点的执行轨迹
"""

from pathlib import Path

from dotenv import load_dotenv

# 必须在 import deep_research_from_scratch 之前加载 .env：
# utils.py / prompts.py 顶层就会构建模型客户端，需要 API key 已就位
# （目录整理后 notebooks 位于 research/ 子目录，.env 在仓库根，需向上三级）
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

import asyncio
import os
import re

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
    get_buffer_string,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field
from typing_extensions import Literal

from deep_research_from_scratch.prompts import (
    transform_messages_into_research_topic_prompt,
    lead_researcher_prompt,
    research_agent_prompt,
    compress_research_system_prompt,
    compress_research_human_message,
    final_report_generation_prompt,
)
from deep_research_from_scratch.state_scope import (
    AgentState,
    AgentInputState,
)
from deep_research_from_scratch.state_research import ResearcherState, ResearcherOutputState
from deep_research_from_scratch.state_multi_agent_supervisor import (
    SupervisorState,
    ConductResearch,
    ResearchComplete,
)
from deep_research_from_scratch.utils import get_today_str, think_tool, tavily_search

# ===== 模型配置 =====

# 智谱 GLM（OpenAI 兼容端点，key 为 .env 中的 ZHIPUAI_API_KEY）
_ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
# glm-5.3-flashx 始终开启思考，不支持 disabled，只支持 low/high/max
_ZHIPU_KWARGS = dict(
    model="openai:glm-5.3-flashx",
    base_url=_ZHIPU_BASE_URL,
    api_key=os.environ["ZHIPUAI_API_KEY"],
    reasoning_effort="low",
)

# scope / supervisor / researcher 共用的对话与决策模型
model = init_chat_model(**_ZHIPU_KWARGS)
# researcher 结果压缩模型
compress_model = init_chat_model(**_ZHIPU_KWARGS)
# 最终报告撰写模型（notebook 5 原为 openai:gpt-4.1, max_tokens=32000）
writer_model = init_chat_model(**_ZHIPU_KWARGS, max_tokens=8192)

# ===== 系统常量（可通过 .env 覆盖，未设置时使用下方默认值）=====


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"[配置] {name}={raw!r} 不是有效整数，使用默认值 {default}")
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


# supervisor 的最大研究轮数（一轮 = 派题 -> 执行 -> 复盘）。设为 2 时只够
# 「先 think_tool 再派题」的模型行为执行一批研究，建议 >= 3
max_researcher_iterations = _env_int("MAX_RESEARCHER_ITERATIONS", 3)
# 单个 researcher 子代理最多执行的 tavily_search 次数（硬上限，达到后强制进入压缩）
max_researcher_searches = _env_int("MAX_RESEARCHER_SEARCHES", 3)
# supervisor 每轮最多并行启动的研究员数量
max_concurrent_researchers = _env_int("MAX_CONCURRENT_RESEARCHERS", 3)
# brief 人工审核的最大打回修订轮数（超过后仅提示警告，不强制中断）
max_brief_review_rounds = _env_int("MAX_BRIEF_REVIEW_ROUNDS", 5)
# 调试开关：True 时简报确认后直接 END，跳过 supervisor/研究阶段，
# 只用于验证协商循环与简报产出质量（脚本层会相应优雅退出并打印最终简报）
brief_only_debug_mode = _env_bool("BRIEF_ONLY_DEBUG_MODE", False)

# ===== 第一部分：scope 阶段（激进合并版：原 clarify + write_brief 合为单节点）=====

# Brief 协商 schema：仿照 ClarifyWithUser 的「全字段必填 + 空串占位」惯例。
# 合并后这是唯一的判断点：首跑请求模糊、或修订反馈模糊时反问澄清，清晰时产出简报
class BriefRevision(BaseModel):
    """Schema for brief scoping/revision: revise directly or ask a clarifying question."""

    need_clarification: bool = Field(
        description="Whether the conversation provides too little direction to produce a high-quality research brief confidently.",
    )
    clarifying_question: str = Field(
        description="One concise question to ask the user when need_clarification is true, otherwise an empty string.",
    )
    research_brief: str = Field(
        description="The complete research brief when need_clarification is false, otherwise an empty string.",
    )


# 合并版指令：移植原 clarify_with_user_instructions 的提问规则 + 简报修订规则
scope_brief_instructions = """

Before producing the research brief, assess whether the conversation provides sufficiently clear direction:
- When in doubt, ALWAYS ask rather than guess: one clarifying question costs the user seconds, but an ambiguity left unasked silently corrupts the entire research brief and wastes a full research cycle. If anything is unclear, or you are not confident you can structure a high-quality research brief from the conversation, ask the user directly — even if a question was asked earlier in this negotiation. However, only ask about details that materially affect the research direction or scope, never minor stylistic preferences.
- If the user's request or latest feedback is vague, contradictory, or lacks key details such that you cannot confidently produce a high-quality research brief, set "need_clarification" to true and ask ONE concise clarifying question in "clarifying_question" (set "research_brief" to "").
- If there are acronyms, abbreviations, or unknown terms, ask the user to clarify.
- Don't ask for unnecessary information, or information the user has already provided. If a clarifying question has already been asked AND answered in the messages history, almost never ask another one about the same point — only ask again if ABSOLUTELY NECESSARY about a DIFFERENT point.
- If the direction is clear, set "need_clarification" to false, produce the complete research brief in "research_brief" (set "clarifying_question" to ""). If a previous brief version appears in the message history, treat it as the base to revise.

Brief format requirements (applies to every version you produce):
- The brief is a DISTILLED SEARCH DIRECTION for a downstream supervisor that delegates a limited budget of parallel research tasks — NOT an outline of the final report, and NEVER a transcription of the user's bullet points. Condense, don't copy: if the user's request is already highly structured, your job is to compress it into its essential research questions.
- Target length: at most ~300 words.
- Structure: (1) one-sentence core question; (2) 3-5 key sub-questions ranked by priority, each a self-contained research direction; (3) one short line of scope constraints and source preferences (e.g. "prefer primary/official sources; flag company claims").
- Do NOT enumerate every detail or sub-topic from the conversation — the full conversation history remains available to later stages. Details like formatting requests, table styles, or term definitions belong to report writing, not to the brief.
- The 3-5 sub-questions must fit a research budget of roughly 6 supervisor rounds × 3 parallel researchers; prioritize what most affects the research outcome and drop the rest.
"""


# 扩展主图状态（src 的 state_scope.py 不宜改动，本地子类扩展）
class FullAgentState(AgentState):
    # 协商轮数计数：修改意见与澄清问答统一累加，用于软成本警告
    negotiation_rounds: int = 0
    # 模型正在等待用户补充（scope_brief 反问 / 澄清问答共用）
    awaiting_clarification: bool = False


async def scope_brief(state: FullAgentState):
    """唯一的简报生成/修订节点：方向清晰则产出简报，模糊则反问澄清.

    合并了原 clarify_with_user 的判断职责与 write_research_brief 的生成职责。
    注：StateGraph 不校验节点入参标注，这里标注 FullAgentState 只是如实描述
    运行时实际收到的全量状态（节点实际只读写其中少数几个键）。
    """
    structured_output_model = model.with_structured_output(BriefRevision)

    response = await structured_output_model.ainvoke([
        HumanMessage(content=transform_messages_into_research_topic_prompt.format(
            messages=get_buffer_string(state.get("messages", [])),
            date=get_today_str()
        ) + scope_brief_instructions)
    ])

    if response.need_clarification:
        # 请求/反馈模糊：不改稿，写入澄清问题后进入 review_brief 等用户补充
        print(f"\n[brief] 方向不够清晰，向用户反问: {response.clarifying_question}")
        return {
            "messages": [AIMessage(content=response.clarifying_question)],
            "awaiting_clarification": True,
        }

    print(f"\n[brief] 研究简报: {response.research_brief}...")

    # 不在此处播种 supervisor_messages（add_messages 只增不减，多轮修订会堆积旧版），
    # 由 review_brief 的 OK 确认处用最终版播种
    return {
        "research_brief": response.research_brief,
        "awaiting_clarification": False,
    }


async def review_brief(state: FullAgentState) -> Command[Literal["scope_brief", "supervisor_subgraph", "__end__"]]:
    """人工审核节点：interrupt() 暂停等用户输入，是全流程唯一的人工入口.

    同时承接两类上游：
    - scope_brief 的澄清问题（awaiting_clarification=True，简报尚未生成）
    - scope_brief 产出的待审简报

    所有出口均为显式 Command 路由（不依赖静态边）：实测 langgraph 1.2.14 中，
    节点声明静态边时 interrupt 恢复后返回的 Command(goto=END) 会被静态边覆盖，
    因此必须删掉静态边、每条路径显式指定去向：
    - OK 类确认 + 简报已生成 -> 播种 supervisor_messages 后进 supervisor
      （调试模式下直接 goto=END 跳过研究阶段）
    - OK 类确认 + 简报未生成 -> 视为「别问了，直接基于已有信息生成简报」打回
    - 其他输入一律作为 [用户补充] 写回 messages，打回 scope_brief
    """
    brief = state.get("research_brief", "")
    rounds = state.get("negotiation_rounds", 0)
    awaiting = state.get("awaiting_clarification", False)

    if awaiting:
        last_message = state.get("messages", [None])[-1]
        question = str(getattr(last_message, "content", "") or "（见对话历史中的澄清问题）")
        interrupt_payload = (
            "模型有一个问题需要你补充说明：\n"
            f"{question}\n\n"
            "请直接输入回答；回复 OK 则要求模型基于已有信息直接生成/继续。"
        )
    else:
        interrupt_payload = (
            "研究简报已生成，请审阅。回复 OK 进入研究阶段；"
            "或直接输入修改意见，我将基于你的反馈修订简报。"
        )

    user_response = str(interrupt(interrupt_payload)).strip()
    is_ok = user_response.lower() in {"ok", "确认", "yes", "y"} or user_response == ""

    if is_ok and brief:
        # 简报已确认：正式模式在此处才把最终版播种给 supervisor（避免多轮修订的旧版堆积）
        if brief_only_debug_mode:
            # 调试模式：确认后直接结束，跳过研究阶段（research_brief 已在 state 中可查）
            print(f"[调试] 简报协商完成（brief_only_debug_mode=True，跳过研究阶段）:\n{brief}")
            return Command(goto=END)
        return Command(
            goto="supervisor_subgraph",
            update={
                "supervisor_messages": [HumanMessage(content=f"{brief}.")],
                "awaiting_clarification": False,
            },
        )

    if is_ok and not brief:
        # clarify 阶段就回 OK：视为「没有更多补充，直接生成简报」
        print("[提示] 简报尚未生成，OK 视为要求直接基于已有信息生成简报")
        return Command(
            goto="scope_brief",
            update={
                "messages": [HumanMessage(content="[用户补充] （没有更多补充，请直接基于已有信息生成研究简报）")],
                "negotiation_rounds": rounds + 1,
                "awaiting_clarification": False,
            },
        )

    if rounds >= max_brief_review_rounds:
        print(f"[警告] 已达最大协商轮数（{max_brief_review_rounds}），继续接受反馈但请注意成本")

    # 澄清回答或修改意见：统一作为用户补充写回，打回 scope_brief。
    # 上一版简报快照只在「首次对新鲜简报提意见」时注入；awaiting 路径的历史里已包含
    new_messages = []
    if not awaiting and brief:
        new_messages.append(AIMessage(content=f"[上一版研究简报]\n{brief}"))
    new_messages.append(HumanMessage(content=f"[用户补充] {user_response}"))

    return Command(
        goto="scope_brief",
        update={
            "messages": new_messages,
            "negotiation_rounds": rounds + 1,
            "awaiting_clarification": False,
        },
    )

# ===== 第二部分：researcher 子代理（与 notebook 2 相同结构）=====

researcher_tools = [tavily_search, think_tool]
researcher_tools_by_name = {tool.name: tool for tool in researcher_tools}
researcher_model_with_tools = model.bind_tools(researcher_tools)


async def researcher_llm_call(state: ResearcherState):
    response = await researcher_model_with_tools.ainvoke(
        [SystemMessage(content=research_agent_prompt.format(date=get_today_str()))]
        + state["researcher_messages"]
    )
    return {"researcher_messages": [response]}


class LocalResearcherState(ResearcherState):
    """本地扩展 researcher 状态（src 的 state_research.py 不宜改动）.

    search_count 累计已执行的 tavily_search 次数，用于硬上限控制；
    think_tool 不计数。TypedDict 类属性默认值由 LangGraph 通道采用（输入缺省时为 0）。
    """
    search_count: int = 0


async def researcher_tool_node(state: LocalResearcherState) -> Command[Literal["llm_call", "compress_research"]]:
    tool_calls = state["researcher_messages"][-1].tool_calls

    observations = []
    search_calls = 0
    for tool_call in tool_calls:
        tool = researcher_tools_by_name[tool_call["name"]]
        if tool_call["name"] == "tavily_search":
            search_calls += 1
        observation = await tool.ainvoke(tool_call["args"])
        observations.append(observation)

    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ) for observation, tool_call in zip(observations, tool_calls)
    ]

    # 先回填本批全部 tool_calls（协议要求完整配对），再按累计搜索数决定去向：
    # 达到上限则强制压缩，不再让模型发起新一轮搜索
    new_total = state.get("search_count", 0) + search_calls
    update = {"researcher_messages": tool_outputs, "search_count": new_total}

    if new_total >= max_researcher_searches:
        print(f"[researcher] 搜索次数达上限（{new_total}/{max_researcher_searches}），强制进入压缩")
        return Command(goto="compress_research", update=update)
    return Command(goto="llm_call", update=update)


async def researcher_compress(state: ResearcherState) -> dict:
    system_message = compress_research_system_prompt.format(date=get_today_str())
    messages = (
        [SystemMessage(content=system_message)]
        + state.get("researcher_messages", [])
        # 修复：src 版遗漏了 format，{research_topic} 占位符会原样发给模型
        + [HumanMessage(content=compress_research_human_message.format(
            research_topic=state.get("research_topic", "")
        ))]
    )
    response = await compress_model.ainvoke(messages)

    raw_notes = [
        str(m.content) for m in filter_messages(
            state["researcher_messages"],
            include_types=["tool", "ai"]
        )
    ]
    return {
        "compressed_research": str(response.content),
        "raw_notes": ["\n".join(raw_notes)]
    }


def researcher_should_continue(state: ResearcherState) -> Literal["tool_node", "compress_research"]:
    last_message = state["researcher_messages"][-1]
    if last_message.tool_calls:
        return "tool_node"
    return "compress_research"


researcher_builder = StateGraph(LocalResearcherState, output_schema=ResearcherOutputState)
researcher_builder.add_node("llm_call", researcher_llm_call)
researcher_builder.add_node("tool_node", researcher_tool_node)
researcher_builder.add_node("compress_research", researcher_compress)
researcher_builder.add_edge(START, "llm_call")
researcher_builder.add_conditional_edges(
    "llm_call",
    researcher_should_continue,
    {"tool_node": "tool_node", "compress_research": "compress_research"},
)
# 注意：tool_node 不声明静态边——它的去向（回 llm_call / 强制压缩）由节点内
# Command(goto=...) 依据累计搜索数动态指定，避免静态边覆盖 Command 的路由
researcher_builder.add_edge("compress_research", END)
researcher_agent = researcher_builder.compile()

# ===== 第三部分：supervisor（与 notebook 4 相同结构）=====

def get_notes_from_tool_calls(messages: list) -> list[str]:
    """从 supervisor 消息历史中提取「研究任务 → 压缩成果」成对块（notes）.

    - 先遍历 AI 消息的 tool_calls，建立 tool_call_id → research_topic 映射
    - 再对 ConductResearch 的 ToolMessage 生成「## 研究任务：X + 成果」结构化块，
      让报告模型能按简报子问题对号入座，并暴露覆盖情况（漏研/重复研）
    - 排除 think_tool 的反思记录：过程性元文本混入 notes 会污染报告输入
    """
    topic_by_call_id = {}
    for msg in messages:
        for tc in getattr(msg, "tool_calls", None) or []:
            if tc["name"] == "ConductResearch":
                topic_by_call_id[tc["id"]] = tc["args"].get("research_topic", "（未记录主题）")

    notes = []
    for tool_msg in filter_messages(messages, include_types="tool"):
        if tool_msg.name == "think_tool":
            continue
        topic = topic_by_call_id.get(tool_msg.tool_call_id, "（未记录主题）")
        notes.append(f"## 研究任务：{topic}\n\n{tool_msg.content}")
    return notes


supervisor_tools_list = [ConductResearch, ResearchComplete, think_tool]
supervisor_model_with_tools = model.bind_tools(supervisor_tools_list)


async def supervisor(state: SupervisorState) -> Command[Literal["supervisor_tools"]]:
    supervisor_messages = state.get("supervisor_messages", [])

    system_message = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=max_concurrent_researchers,
        max_researcher_iterations=max_researcher_iterations
    )
    messages = [SystemMessage(content=system_message)] + supervisor_messages

    response = await supervisor_model_with_tools.ainvoke(messages)

    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1
        }
    )


async def supervisor_tools(state: SupervisorState) -> Command[Literal["supervisor", "__end__"]]:
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]

    tool_calls = most_recent_message.tool_calls
    conduct_research_calls = [
        tc for tc in tool_calls if tc["name"] == "ConductResearch"
    ]
    research_complete = any(
        tool_call["name"] == "ResearchComplete" for tool_call in tool_calls
    )
    exceeded_iterations = research_iterations >= max_researcher_iterations

    # 退出条件：无工具调用 / 显式 ResearchComplete / 迭代超限且本轮没有待执行的研究派题。
    # 关键修复：退出判断只对「没有待执行研究动作」的轮次生效——若模型已派出
    # ConductResearch，必须先执行完毕、下一轮再退出；否则 think_tool 思考轮会
    # 消耗掉迭代预算，导致已派出的研究永远不执行，notes 为空，报告阶段拒绝生成。
    should_end = (
        not tool_calls
        or (research_complete and not conduct_research_calls)
        or (exceeded_iterations and not conduct_research_calls)
    )

    if should_end:
        # 回填本轮所有待答复的 tool_calls（think_tool / ResearchComplete），
        # 保证 tool_call 协议配对完整，再携带 notes 结束
        closing_messages = []
        for tool_call in tool_calls:
            content = (
                "Research complete acknowledged."
                if tool_call["name"] == "ResearchComplete"
                else str(think_tool.invoke(tool_call["args"]))
            )
            closing_messages.append(
                ToolMessage(
                    content=content,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        update = {
            "notes": get_notes_from_tool_calls(supervisor_messages + closing_messages),
            "research_brief": state.get("research_brief", ""),
        }
        if closing_messages:
            update["supervisor_messages"] = closing_messages
        return Command(goto=END, update=update)

    # 执行路径：think_tool 同步执行 + ConductResearch 并行派发（不受退出检查拦截）
    tool_messages = []
    all_raw_notes = []
    try:
        think_tool_calls = [
            tc for tc in tool_calls if tc["name"] == "think_tool"
        ]

        # think_tool：同步执行
        for tool_call in think_tool_calls:
            observation = think_tool.invoke(tool_call["args"])
            tool_messages.append(
                ToolMessage(
                    content=observation,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                )
            )

        # ConductResearch：并行启动 researcher 子代理
        if conduct_research_calls:
            print(f"\n[子代理] 并行启动 {len(conduct_research_calls)} 个研究员:")
            for tc in conduct_research_calls:
                print(f"  -> 研究主题: {tc['args']['research_topic'][:80]}...")

            coros = [
                researcher_agent.ainvoke({
                    "researcher_messages": [
                        HumanMessage(content=tool_call["args"]["research_topic"])
                    ],
                    "research_topic": tool_call["args"]["research_topic"]
                })
                for tool_call in conduct_research_calls
            ]

            tool_results = await asyncio.gather(*coros)

            research_tool_messages = [
                ToolMessage(
                    content=result.get("compressed_research", "Error synthesizing research report"),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                ) for result, tool_call in zip(tool_results, conduct_research_calls)
            ]
            tool_messages.extend(research_tool_messages)

            all_raw_notes = [
                "\n".join(result.get("raw_notes", []))
                for result in tool_results
            ]

            print(f"[子代理] 全部研究员完成，共回收 {len(tool_results)} 份压缩成果")

    except Exception as e:
        print(f"Error in supervisor tools: {e}")
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", "")
            }
        )

    return Command(
        goto="supervisor",
        update={
            "supervisor_messages": tool_messages,
            "raw_notes": all_raw_notes
        }
    )

# supervisor 子图（作为整体流程的一个节点嵌入）
supervisor_builder = StateGraph(SupervisorState)
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)
supervisor_builder.add_edge(START, "supervisor")
supervisor_agent = supervisor_builder.compile()

# ===== 第四部分：最终报告生成（notebook 5 新增）=====

async def final_report_generation(state: AgentState):
    """汇总所有研究笔记（notes），生成结构化最终报告."""
    notes = state.get("notes", [])

    findings = "\n".join(notes)

    final_report_prompt = final_report_generation_prompt.format(
        research_brief=state.get("research_brief", ""),
        findings=findings,
        date=get_today_str()
    )

    final_report = await writer_model.ainvoke([HumanMessage(content=final_report_prompt)])

    return {
        "final_report": final_report.content,
        "messages": [AIMessage(content="Here is the final report: " + final_report.content)],
    }

# ===== 图构建：scope_brief 协商 -> supervisor -> final report =====

deep_researcher_builder = StateGraph(FullAgentState, input_schema=AgentInputState)

deep_researcher_builder.add_node("scope_brief", scope_brief)
deep_researcher_builder.add_node("review_brief", review_brief)
deep_researcher_builder.add_node("supervisor_subgraph", supervisor_agent)
deep_researcher_builder.add_node("final_report_generation", final_report_generation)

deep_researcher_builder.add_edge(START, "scope_brief")
deep_researcher_builder.add_edge("scope_brief", "review_brief")
# 注意：review_brief 不声明静态边——它的所有出口（OK 进 supervisor / 打回 scope_brief /
# 调试跳 END）都由节点内的显式 Command(goto=...) 动态指定，见 review_brief docstring
deep_researcher_builder.add_edge("supervisor_subgraph", "final_report_generation")
deep_researcher_builder.add_edge("final_report_generation", END)

# 复杂多智能体工作流步数多，必须提高 recursion_limit（默认 25 不够）
checkpointer = InMemorySaver()
full_agent = deep_researcher_builder.compile(checkpointer=checkpointer)

# ===== 运行：交互式多轮对话 + astream 节点轨迹 =====

DEFAULT_QUESTION = "Compare Gemini to OpenAI Deep Research agents."


def print_node_update(node_name: str, node_update: dict):
    """打印单个节点的状态更新摘要."""
    print(f"\n{'=' * 70}\n[节点] {node_name}")
    for key, value in node_update.items():
        if key == "messages":
            for m in value:
                mtype = m.__class__.__name__
                if getattr(m, "tool_calls", None):
                    print(f"  AI 消息（{len(m.tool_calls)} 个工具调用）:")
                    for tc in m.tool_calls:
                        print(f"    -> {tc['name']}  args={str(tc['args'])[:120]}")
                elif mtype == "ToolMessage":
                    preview = str(m.content).replace("\n", " ")[:200]
                    print(f"  工具结果 {m.name}（{len(str(m.content))} 字符）: {preview}...")
                else:
                    print(f"  {mtype}: {str(m.content)[:300]}")
        elif key == "supervisor_messages":
            for m in value:
                if getattr(m, "tool_calls", None):
                    print(f"  Supervisor AI 消息（{len(m.tool_calls)} 个工具调用）:")
                    for tc in m.tool_calls:
                        print(f"    -> {tc['name']}  args={str(tc['args'])[:120]}")
                else:
                    print(f"  {m.__class__.__name__}: {str(m.content)[:200]}")
        elif key == "final_report":
            print(f"  final_report:（{len(str(value))} 字符，最后统一展示）")
        else:
            preview = str(value).replace("\n", " ")[:200]
            print(f"  {key}: {preview}{'...' if len(preview) >= 200 else ''}")


async def stream_run(graph_input, thread: dict) -> list:
    """以 astream 方式跑一段图执行，打印节点更新，返回捕获到的 interrupt 列表.

    graph_input 既可以是正常输入（{"messages": [...]}），也可以是
    Command(resume=...) 续跑指令。
    """
    interrupts = []
    async for update in full_agent.astream(graph_input, config=thread, stream_mode="updates"):
        for node_name, node_update in update.items():
            if node_name == "__interrupt__":
                for intr in node_update:
                    interrupts.append(intr)
                    print(f"\n{'!' * 70}\n[中断] 图已暂停，等待人工输入")
                    print(f"{intr.value}")
                    print(f"{'!' * 70}")
            elif node_update:
                print_node_update(node_name, node_update)
    return interrupts


async def run_full_agent():
    thread = {"configurable": {"thread_id": "1", "recursion_limit": 50}}

    print("=== Deep Research 全流程（scope_brief 协商 -> supervisor -> report）===")
    user_input = input("🧑 你（回车使用默认问题）: ").strip() or DEFAULT_QUESTION
    print(f"🧑 你: {user_input}")

    graph_input = {"messages": [HumanMessage(content=user_input)]}

    # 统一循环：图只会因 interrupt 暂停（等人工输入）或产出最终报告而结束
    while True:
        interrupts = await stream_run(graph_input, thread)
        snapshot = await full_agent.aget_state(thread)
        values = snapshot.values or {}

        if values.get("final_report"):
            break

        if interrupts:
            review = input("🧑 回复（OK 继续 / 直接输入意见或回答）: ").strip() or "OK"
            print(f"🧑 你: {review}")
            graph_input = Command(resume=review)
        else:
            if brief_only_debug_mode and values.get("research_brief"):
                # 调试模式：图因简报确认而正常 END，展示简报后优雅退出
                print(f"\n{'#' * 70}\n# [调试] 协商循环结束，最终研究简报\n{'#' * 70}\n")
                print(values["research_brief"])
                return
            print("[错误] 图意外结束且无最终报告，终止运行")
            return

    # 最终报告
    print(f"\n{'#' * 70}\n# 最终报告\n{'#' * 70}\n")
    print(values["final_report"])

    # 可选：保存到本地文件
    # 报告文件名取自本次任务的研究简报（清洗 Windows 非法字符 + 截断），
    # 多次调研各自落盘互不覆盖
    brief_slug = re.sub(r'[\\/:*?"<>|\r\n]+', " ", values.get("research_brief", "")).strip()
    brief_slug = (brief_slug[:60].strip().rstrip(". ") or "report")
    out_path = Path(__file__).resolve().parent / f"final_report_{brief_slug}.md"
    out_path.write_text(values["final_report"], encoding="utf-8")
    print(f"\n[完成] 报告已保存至: {out_path}")


if __name__ == "__main__":
    asyncio.run(run_full_agent())
