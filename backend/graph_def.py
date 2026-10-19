"""深度研究图的可导入构建模块（从 notebooks/5_full_agent copy.py 抽取）.

职责：仅负责图的定义与编译，去掉 CLI 交互部分（input 循环、报告落盘），
由 backend/runner.py 以「astream -> 捕获 interrupt -> Command(resume)」循环驱动。

与脚本版的差异：
- checkpointer 由 build_graph(checkpointer) 注入（服务端用 AsyncSqliteSaver，
  支持进程重启后从 interrupt 处恢复会话）
- max_researcher_iterations 默认调整为 2：设为 1 时 supervisor 第一轮即满足
  退出条件，一个研究员都不会派出
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

# 必须在 import deep_research_from_scratch 之前加载 .env：
# utils.py / prompts.py 顶层就会构建模型客户端，需要 API key 已就位
_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")

# pyproject 位于 research/ 子目录且包未 editable 安装时，直接把 src 目录加入导入路径
_SRC = str(_REPO_ROOT / "research" / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import asyncio
import os

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
    get_buffer_string,
)
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
# 最终报告撰写模型
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
# 调试开关：True 时简报确认后直接 END，跳过 supervisor/研究阶段
brief_only_debug_mode = _env_bool("BRIEF_ONLY_DEBUG_MODE", False)

# ===== 第一部分：scope 阶段（原 clarify + write_brief 合为单节点）=====


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


class FullAgentState(AgentState):
    """扩展主图状态：增加协商轮数与澄清等待标记."""

    negotiation_rounds: int = 0
    awaiting_clarification: bool = False


async def scope_brief(state: FullAgentState):
    """唯一的简报生成/修订节点：方向清晰则产出简报，模糊则反问澄清."""
    structured_output_model = model.with_structured_output(BriefRevision)

    response = await structured_output_model.ainvoke([
        HumanMessage(content=transform_messages_into_research_topic_prompt.format(
            messages=get_buffer_string(state.get("messages", [])),
            date=get_today_str()
        ) + scope_brief_instructions)
    ])

    if response.need_clarification:
        print(f"[brief] 方向不够清晰，向用户反问: {response.clarifying_question}")
        return {
            "messages": [AIMessage(content=response.clarifying_question)],
            "awaiting_clarification": True,
        }

    print(f"[brief] 研究简报已生成（{len(response.research_brief)} 字符）")
    return {
        "research_brief": response.research_brief,
        "awaiting_clarification": False,
    }


async def review_brief(state: FullAgentState) -> Command[Literal["scope_brief", "supervisor_subgraph", "__end__"]]:
    """人工审核节点：interrupt() 暂停等用户输入，是全流程唯一的人工入口.

    所有出口均为显式 Command 路由（不依赖静态边）：interrupt 恢复后返回的
    Command(goto=END) 会被静态边覆盖，因此每条路径显式指定去向。
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
        if brief_only_debug_mode:
            print(f"[调试] 简报协商完成（brief_only_debug_mode=True，跳过研究阶段）")
            return Command(goto=END)
        return Command(
            goto="supervisor_subgraph",
            update={
                "supervisor_messages": [HumanMessage(content=f"{brief}.")],
                "awaiting_clarification": False,
            },
        )

    if is_ok and not brief:
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

# ===== 第二部分：researcher 子代理 =====

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
    """本地扩展 researcher 状态：search_count 累计 tavily_search 硬上限计数."""

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
researcher_builder.add_edge("compress_research", END)
researcher_agent = researcher_builder.compile()

# ===== 第三部分：supervisor =====


def get_notes_from_tool_calls(messages: list) -> list[str]:
    """从 supervisor 消息历史中提取「研究任务 → 压缩成果」成对块（notes）.

    排除 think_tool 的反思记录，避免过程性元文本污染报告输入。
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

    # 终止条件：无工具调用 / 显式 ResearchComplete / 迭代超限且本轮没有待执行的研究派题。
    # 关键：退出判断只对「没有待执行研究动作」的轮次生效——若模型已派出
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

        for tool_call in think_tool_calls:
            observation = think_tool.invoke(tool_call["args"])
            tool_messages.append(
                ToolMessage(
                    content=observation,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                )
            )

        if conduct_research_calls:
            print(f"[子代理] 并行启动 {len(conduct_research_calls)} 个研究员")
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


supervisor_builder = StateGraph(SupervisorState)
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)
supervisor_builder.add_edge(START, "supervisor")
supervisor_agent = supervisor_builder.compile()

# ===== 第四部分：最终报告生成 =====


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

# ===== 图构建 =====


def build_graph(checkpointer):
    """编译完整研究图：scope_brief 协商 -> supervisor -> final report.

    checkpointer 由调用方注入（服务端传入 AsyncSqliteSaver），
    同一 thread_id 的会话可在进程重启后从 interrupt 处恢复。
    """
    builder = StateGraph(FullAgentState, input_schema=AgentInputState)

    builder.add_node("scope_brief", scope_brief)
    builder.add_node("review_brief", review_brief)
    builder.add_node("supervisor_subgraph", supervisor_agent)
    builder.add_node("final_report_generation", final_report_generation)

    builder.add_edge(START, "scope_brief")
    builder.add_edge("scope_brief", "review_brief")
    builder.add_edge("supervisor_subgraph", "final_report_generation")
    builder.add_edge("final_report_generation", END)

    return builder.compile(checkpointer=checkpointer)
