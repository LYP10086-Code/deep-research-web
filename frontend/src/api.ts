export interface Session {
  id: string
  title: string
  status: 'running' | 'waiting_review' | 'completed' | 'failed'
  brief?: string
  created_at: string
  updated_at: string
}

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  created_at: string
}

export interface ReportMeta {
  id: string
  session_id: string
  title: string
  generated_at: string
}

export interface Report extends ReportMeta {
  brief: string
  content: string
}

async function jfetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json; charset=utf-8' },
    ...init,
  })
  if (!res.ok) {
    const body = await res.text()
    throw new Error(`${res.status}: ${body}`)
  }
  return res.json() as Promise<T>
}

export const api = {
  createSession: (question: string) =>
    jfetch<Session>('/api/sessions', { method: 'POST', body: JSON.stringify({ question }) }),

  getSessions: () => jfetch<Session[]>('/api/sessions'),

  getSession: (id: string) => jfetch<Session>(`/api/sessions/${id}`),

  getMessages: (id: string) => jfetch<ChatMessage[]>(`/api/sessions/${id}/messages`),

  resume: (id: string, feedback: string) =>
    jfetch<{ ok: boolean }>(`/api/sessions/${id}/resume`, {
      method: 'POST',
      body: JSON.stringify({ feedback }),
    }),

  getReports: () => jfetch<ReportMeta[]>('/api/reports'),

  getReport: (id: string) => jfetch<Report>(`/api/reports/${id}`),

  getActive: () => jfetch<{ active: number }>('/api/stats/active'),
}

export interface SseHandlers {
  onStatus?: (status: Session['status'], interrupt: string | null) => void
  onNodeUpdate?: (node: string) => void
  onMessage?: (msg: { role: 'user' | 'assistant'; content: string }) => void
  onBrief?: (brief: string) => void
  onInterrupt?: (payload: { payload: string; brief: string }) => void
  onReportDone?: (data: { report_id: string }) => void
  onError?: (data: { message: string }) => void
}

/** 订阅某会话的 SSE 事件流，返回关闭函数 */
export function subscribeEvents(sessionId: string, handlers: SseHandlers): () => void {
  const es = new EventSource(`/api/sessions/${sessionId}/events`)

  es.addEventListener('status', (e) => {
    const d = JSON.parse((e as MessageEvent).data)
    handlers.onStatus?.(d.status, d.interrupt)
  })
  es.addEventListener('node_update', (e) => {
    handlers.onNodeUpdate?.(JSON.parse((e as MessageEvent).data).node)
  })
  es.addEventListener('message', (e) => {
    handlers.onMessage?.(JSON.parse((e as MessageEvent).data))
  })
  es.addEventListener('brief', (e) => {
    handlers.onBrief?.(JSON.parse((e as MessageEvent).data).brief)
  })
  es.addEventListener('interrupt', (e) => {
    handlers.onInterrupt?.(JSON.parse((e as MessageEvent).data))
  })
  es.addEventListener('report_done', (e) => {
    handlers.onReportDone?.(JSON.parse((e as MessageEvent).data))
  })
  es.addEventListener('error', (e) => {
    // EventSource 自身的连接错误（如服务重启）也会走 error；data 存在才是业务错误
    if ((e as MessageEvent).data) {
      handlers.onError?.(JSON.parse((e as MessageEvent).data))
    }
  })

  return () => es.close()
}

/** 节点名 -> 前端展示文案 */
export const NODE_LABELS: Record<string, string> = {
  scope_brief: '生成/修订研究简报',
  review_brief: '等待人工审阅',
  supervisor_subgraph: '多智能体并行研究中',
  final_report_generation: '撰写最终报告',
}
