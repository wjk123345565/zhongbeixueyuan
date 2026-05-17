from src.state import AgentState

def task_router(state: AgentState) -> str:
    """
    中央调度枢纽：根据任务队列的状态决定图的走向
    """
    plan = state.get("plan", [])
    
    # ==========================================
    # 场景 1：任务清单空了，去写最终报告
    # ==========================================
    if len(plan) == 0:
        print("🔀 调度中心：所有任务已完成，进入最后汇总阶段...")
        return "synthesizer"
    
    # ==========================================
    # 场景 2：清单里还有任务，派发当前的第一个任务
    # ==========================================
    current_task = plan[0]
    print("🔀 调度中心：正在派发当前任务 -> [" + current_task + "]")
    
    # 因为我们在 Planner 的 Prompt 中明确规定了要带上"SQL专员"或"RAG专员"的字眼
    # 所以这里用非常轻量、极速的关键词匹配就能精准路由，不需要再调一次大模型
    if "SQL" in current_task or "分数" in current_task or "数据" in current_task:
        return "data_analyst"
        
    elif "RAG" in current_task or "政策" in current_task or "简章" in current_task:
        return "rag_agent"
        
    else:
        # 兜底，交给闲聊接待
        return "chat_agent"