from src.state import AgentState
from typing import List

def task_router(state: AgentState) -> str:
    """
    中央调度中心 (流水线分配员)
    
    改进要点：
    1. 优先级调整：优先识别 RAG/知识类任务，防止其被通用的“查询”关键词拦截。
    2. 关键词排他性：在 SQL 匹配中排除掉“课程”、“简章”等 RAG 专属词汇。
    3. 标签化增强：支持 [KNOWLEDGE] 和 [DATA] 强制标签。
    """
    
    # 获取当前任务清单
    plan: List[str] = state.get("plan", [])
    
    # 1. 任务清单检查：如果队列为空，进入汇总阶段
    if not plan:
        print("🔀 [调度中心] 任务队列为空，进入最后汇总阶段...")
        return "synthesizer"
    
    # 2. 获取当前排在最前面的任务，并进行规范化处理
    current_task = plan[0].upper().strip()
    
    # 3. 关键字分派逻辑（注意：顺序决定优先级）

    # --- A. 政策/知识检索类 (优先判断，防止误入 SQL) ---
    # 增加“课程”、“专业介绍”等关键词，并包含 [KNOWLEDGE] 标签
    rag_keywords = [
        "RAG", "政策", "知识库", "简介", "怎么样", "办", "流程", 
        "课程", "学费", "简章", "性质", "要求", "[KNOWLEDGE]"
    ]
    if any(k in current_task for k in rag_keywords):
        print("🔀 [调度中心] 匹配到知识检索任务 -> [RAG 专员]: " + current_task)
        return "rag_agent"

    # --- B. 数据查询类 (SQL 专家) ---
    # 改进：仅在不包含 RAG 核心词的情况下，才根据“分数”、“数据”派发给 SQL
    data_keywords = ["SQL", "数据", "分数", "录取", "是多少", "查询", "最低分", "位次", "[DATA]"]
    if any(k in current_task for k in data_keywords):
        # 再次确认不是课程类咨询，防止“查询音乐专业课程”误入
        if not any(k in current_task for k in ["课程", "学费", "介绍"]):
            print("🔀 [调度中心] 匹配到精确数据任务 -> [SQL 专员]: " + current_task)
            return "data_analyst"
        
    # --- C. 日常寒暄/闲聊类 ---
    chat_keywords = ["闲聊", "CHAT", "你好", "您好", "再见", "谢谢", "你是谁"]
    if any(k in current_task for k in chat_keywords):
        print("🔀 [调度中心] 匹配到日常接待任务...")
        return "chat_agent"
        
    # 4. 兜底保护逻辑
    else:
        print("⚠️ [调度中心] 无法识别的任务类型: " + current_task + "，强制进入汇总。")
        return "synthesizer"