from langgraph.graph import StateGraph, END
from src.state import AgentState

# ==========================================
# 1. 导入所有大脑分区 (Nodes)
# ==========================================
from src.nodes.planner import planner_node
from src.nodes.data_analyst import sql_generator_node, sql_executor_node
# 💥 导入高精度 RAG 专员节点
from src.nodes.rag_expert import rag_expert_node 
from src.nodes.chat import chat_agent_node
from src.nodes.synthesizer import synthesizer_node

# ==========================================
# 2. 导入路由边逻辑 (Edges)
# ==========================================
from src.nodes.router import task_router # 确保路径与你的 router.py 一致

# ==========================================
# 3. 内部路由：SQL 执行后的走向判定
# ==========================================
def after_sql_execution(state: AgentState) -> str:
    """
    检查 SQL 执行结果：
    1. 如果有报错，打回给 sql_generator 重新编写 SQL。
    2. 如果执行成功（或重试次数耗尽），则返回 task_router 决定下一步去向。
    """
    if state.get("sql_error"):
        print("🔀 [路由] 发现 SQL 报错，打回给 Generator 重做...")
        return "sql_generator"
    else:
        print("🔀 [路由] SQL 环节结束，返回中央调度中心检查剩余任务...")
        # 改进：调用中央调度器，根据 state["plan"] 决定是去 RAG、继续 SQL 还是汇总
        return task_router(state)

# ==========================================
# 4. 构建 LangGraph 状态图
# ==========================================
workflow = StateGraph(AgentState)

# 注册所有节点
workflow.add_node("planner", planner_node)
workflow.add_node("sql_generator", sql_generator_node)
workflow.add_node("sql_executor", sql_executor_node)
workflow.add_node("rag_expert", rag_expert_node) 
workflow.add_node("chat_agent", chat_agent_node)
workflow.add_node("synthesizer", synthesizer_node)

# 起始点：用户提问 -> 找 Planner 拆解任务清单
workflow.set_entry_point("planner")

# ==========================================
# 5. 编排工作流流转 (Edges)
# ==========================================

# 核心调度 1：Planner 拆解完，交给 task_router 分派任务
workflow.add_conditional_edges(
    "planner", 
    task_router, 
    {
        "data_analyst": "sql_generator", # 查数据库任务
        "rag_agent": "rag_expert",       # 查政策文件任务
        "chat_agent": "chat_agent",      # 闲聊任务
        "synthesizer": "synthesizer"     # 直接汇总
    }
)

# 核心调度 2：RAG 专员干完活，必须回到 task_router 重新排队检查是否还有其他任务
workflow.add_conditional_edges(
    "rag_expert",
    task_router,
    {
        "data_analyst": "sql_generator",
        "rag_agent": "rag_expert",       # 若还有连续的 RAG 任务则循环
        "synthesizer": "synthesizer"     # 任务完成，汇总报告
    }
)

# 【重点】SQL 反思闭环：Generator 生成 -> Executor 执行
workflow.add_edge("sql_generator", "sql_executor")

# 核心调度 3：执行完后，利用 after_sql_execution 判断是重做还是接下一个任务
workflow.add_conditional_edges(
    "sql_executor",
    after_sql_execution,
    {
        "sql_generator": "sql_generator", # 报错打回循环 🔄
        "data_analyst": "sql_generator",  # task_router 返回的映射
        "rag_agent": "rag_expert",        # SQL 完事后如果还有 RAG 任务
        "synthesizer": "synthesizer"      # SQL 完事后直接汇总
    }
)

# 最终汇总节点和闲聊节点跑完后，流程正式结束
workflow.add_edge("synthesizer", END)
workflow.add_edge("chat_agent", END)

# 编译应用
app = workflow.compile()