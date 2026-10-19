"""会话任务执行器：每个会话一个 asyncio 后台任务驱动 LangGraph 图.

核心循环（与 5_full_agent copy.py 的 run_full_agent 一致，去掉 CLI）：
  astream 执行 -> 捕获 __interrupt__ -> 状态置 waiting_review 并广播 SSE
  -> 等待 /resume 投递的反馈 -> Command(resume) 续跑 -> 直至产出最终报告。

SSE 事件通过每会话的订阅者队列（asyncio.Queue）扇出，前端每个窗口
持有一条 SSE 长连接；同一会话多个窗口同时打开也能同步收事件。
"""

import asyncio
import re

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from backend import db

# 报告正文会以一条超长 AIMessage 回流对话历史，聊天流中不重复展示
_REPORT_MSG_PREFIXES = ("Here is the final report",)
# 图内部注入的过程性消息，不作为聊天记录展示
_INTERNAL_MSG_PREFIXES = ("[上一版研究简报]",)
_USER_SUPPLEMENT_PREFIX = "[用户补充]"

# stream_mode="updates" 只在节点完成时产出更新。前端需要的是「正在进行哪个阶段」，
# 因此节点完成后广播它即将进入的下一阶段，而不是已完成的节点本身——否则续跑后
# 整个 supervisor_subgraph 执行期间会一直停留在 review_brief 的旧标签上
_NEXT_STAGE = {
    "scope_brief": "review_brief",
    "review_brief": "supervisor_subgraph",
    "supervisor_subgraph": "final_report_generation",
}


class SessionRuntime:
    """单个会话的进程内运行时状态（会话结束后仍保留，供 SSE 补发状态）."""

    def __init__(self) -> None:
        self.resume_queue: asyncio.Queue = asyncio.Queue()
        self.subscribers: set[asyncio.Queue] = set()
        self.task: asyncio.Task | None = None
        self.last_interrupt: str | None = None


class SessionManager:
    """管理所有会话：启动 / 恢复 / 续跑 / SSE 订阅."""

    def __init__(self, graph) -> None:
        self.graph = graph
        self.runtimes: dict[str, SessionRuntime] = {}

    # ===== 生命周期 =====

    def start_session(self, question: str) -> str:
        title = re.sub(r"\s+", " ", question).strip()[:40] or "未命名研究"
        session_id = db.create_session(title)
        db.add_message(session_id, "user", question)
        rt = SessionRuntime()
        self.runtimes[session_id] = rt
        rt.task = asyncio.create_task(self._run(session_id, {"messages": [HumanMessage(content=question)]}))
        return session_id

    def resume(self, session_id: str, feedback: str) -> None:
        """向等待中的会话投递人工审核反馈（OK / 澄清回答 / 修改意见）.

        反馈先作为用户消息落库并广播（保留完整对话记录），再驱动图续跑：
        进程内存活：直接投递 resume 队列；
        服务重启后恢复：图状态已由 checkpointer 持久化，重新拉起后台任务续跑。
        """
        feedback = feedback.strip()
        db.add_message(session_id, "user", feedback)
        self._broadcast(session_id, "message", {"role": "user", "content": feedback})
        rt = self.runtimes.get(session_id)
        if rt is not None and rt.task is not None and not rt.task.done():
            rt.resume_queue.put_nowait(feedback)
            return
        db.set_session_status(session_id, "running")
        rt = self.runtimes.setdefault(session_id, SessionRuntime())
        rt.task = asyncio.create_task(self._run(session_id, Command(resume=feedback)))

    def recover_on_startup(self) -> None:
        """服务重启后的会话恢复：running 视为被打断置 failed，waiting_review 可续跑."""
        for session in db.list_sessions():
            if session["status"] == "running":
                db.set_session_status(session["id"], "failed")

    # ===== SSE 订阅 =====

    def subscribe(self, session_id: str) -> asyncio.Queue:
        rt = self.runtimes.setdefault(session_id, SessionRuntime())
        queue: asyncio.Queue = asyncio.Queue()
        rt.subscribers.add(queue)
        # 连接建立即补发当前状态快照，前端刷新/后进入窗口能立刻对齐
        session = db.get_session(session_id)
        queue.put_nowait({
            "event": "status",
            "data": {
                "status": session["status"] if session else "unknown",
                "interrupt": rt.last_interrupt,
            },
        })
        return queue

    def unsubscribe(self, session_id: str, queue: asyncio.Queue) -> None:
        rt = self.runtimes.get(session_id)
        if rt is not None:
            rt.subscribers.discard(queue)

    def _broadcast(self, session_id: str, event: str, data: dict) -> None:
        rt = self.runtimes.get(session_id)
        if rt is None:
            return
        payload = {"event": event, "data": data}
        for queue in list(rt.subscribers):
            queue.put_nowait(payload)

    # ===== 执行循环 =====

    async def _run(self, session_id: str, graph_input) -> None:
        rt = self.runtimes.setdefault(session_id, SessionRuntime())
        config = {"configurable": {"thread_id": session_id, "recursion_limit": 50}}
        if not isinstance(graph_input, Command):
            # 新会话从 scope_brief 起跑，先广播起始阶段
            self._broadcast(session_id, "node_update", {"node": "scope_brief"})
        try:
            while True:
                interrupts = await self._stream_once(session_id, graph_input, config)
                snapshot = await self.graph.aget_state(config)
                values = snapshot.values or {}

                if values.get("final_report"):
                    self._finish(session_id, values)
                    return

                if interrupts:
                    rt.last_interrupt = str(interrupts[0].value)
                    # 简报以助手消息进入对话流（待确认/每次修订各追加一条新版），
                    # 用户的 OK / 修改意见由 resume() 作为用户消息紧随其后
                    brief = str(values.get("research_brief", "") or "")
                    if brief:
                        brief_msg = f"【研究简报】\n\n{brief}"
                        db.add_message(session_id, "assistant", brief_msg)
                        self._broadcast(session_id, "message", {"role": "assistant", "content": brief_msg})
                    db.set_session_status(session_id, "waiting_review")
                    self._broadcast(session_id, "interrupt", {
                        "payload": rt.last_interrupt,
                        "brief": values.get("research_brief", ""),
                    })
                    feedback = await rt.resume_queue.get()
                    rt.last_interrupt = None
                    db.set_session_status(session_id, "running")
                    graph_input = Command(resume=feedback)
                    continue

                db.set_session_status(session_id, "failed")
                print(f"[runner] 会话 {session_id} 意外结束：无 interrupt 也无最终报告")
                self._broadcast(session_id, "error", {"message": "图意外结束且无最终报告"})
                return
        except asyncio.CancelledError:
            db.set_session_status(session_id, "failed")
            raise
        except Exception as exc:
            db.set_session_status(session_id, "failed")
            print(f"[runner] 会话 {session_id} 异常失败: {type(exc).__name__}: {exc}")
            self._broadcast(session_id, "error", {"message": f"{type(exc).__name__}: {exc}"})

    async def _stream_once(self, session_id: str, graph_input, config) -> list:
        """跑一段图执行，转发节点进度与聊天消息增量，返回捕获的 interrupt 列表."""
        interrupts = []
        async for update in self.graph.astream(graph_input, config=config, stream_mode="updates"):
            if "__interrupt__" in update:
                interrupts.extend(update["__interrupt__"])
                continue
            for node_name, node_update in update.items():
                if not node_update:
                    continue
                self._broadcast(session_id, "node_update", {"node": _NEXT_STAGE.get(node_name, node_name)})
                if isinstance(node_update, dict):
                    # 最新版研究简报：每次修订覆盖落库（确认后不再变化），实时推给前端
                    brief = node_update.get("research_brief")
                    if brief:
                        db.update_session_brief(session_id, str(brief))
                        self._broadcast(session_id, "brief", {"brief": str(brief)})
                    if node_update.get("messages"):
                        self._sync_delta_messages(session_id, node_update["messages"])
        return interrupts

    def _sync_delta_messages(self, session_id: str, messages: list) -> None:
        """把本轮节点新增消息中的「人类可读聊天消息」落库并广播.

        过滤规则：ToolMessage / 带 tool_calls 的 AI 消息 / 内部注入消息全部跳过；
        图内部注入的 [用户补充] 不再落库——该回复已由 resume() 作为用户消息记录，
        重复落库会让对话流出现两条相同的用户消息。
        """
        for msg in messages:
            if isinstance(msg, HumanMessage):
                content = str(msg.content).strip()
                if content.startswith(_USER_SUPPLEMENT_PREFIX):
                    # 用户的真实回复已由 resume() 落库，跳过图内部注入的副本
                    continue
                if not content:
                    continue
                db.add_message(session_id, "user", content)
                self._broadcast(session_id, "message", {"role": "user", "content": content})
            elif isinstance(msg, AIMessage):
                if getattr(msg, "tool_calls", None):
                    continue
                content = str(msg.content).strip()
                if not content or content.startswith(_INTERNAL_MSG_PREFIXES) or content.startswith(_REPORT_MSG_PREFIXES):
                    continue
                db.add_message(session_id, "assistant", content)
                self._broadcast(session_id, "message", {"role": "assistant", "content": content})

    def _finish(self, session_id: str, values: dict) -> None:
        """图正常结束且产出最终报告：报告作为对话消息落库，报告元数据入库，会话置 completed."""
        report = str(values["final_report"])
        brief = str(values.get("research_brief", ""))
        # 最终报告作为最后一条对话消息，保证时间线完整：...简报 -> OK -> 报告
        report_msg = f"【最终研究报告】\n\n{report}"
        db.add_message(session_id, "assistant", report_msg)
        self._broadcast(session_id, "message", {"role": "assistant", "content": report_msg})
        session = db.get_session(session_id)
        title = session["title"] if session else "研究报告"
        report_id = db.save_report(session_id, title, brief, report)
        db.set_session_status(session_id, "completed")
        self._broadcast(session_id, "report_done", {"report_id": report_id})
