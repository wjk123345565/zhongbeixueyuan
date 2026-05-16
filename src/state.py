from typing import List, Annotated, TypedDict, Optional, Union
from langchain_core.messages import BaseMessage
import operator

def manage_messages(old_messages: List[BaseMessage], new_messages: Union[BaseMessage, List[BaseMessage]]):
    """
    改进：自定义消息 Reducer
    1. 支持单条消息或消息列表的累加。
    2. 自动修剪上下文：仅保留最近的 12 条消息，防止 Token 溢出并降低推理成本。
    """
    if isinstance(new_messages, BaseMessage):
        combined = old_messages + [new_messages]
    else:
        combined = old_messages + new_messages
    
    # 保持最近的 12 条记录（约 6 轮对话），确保 Agent 拥有足够的短期记忆
    return combined[-12:]

class AgentState(TypedDict):
    """
    中北学院招生 Agent 的全局状态定义
    """
    # 聊天记录：使用自定义的 manage_messages 进行长度控制
    messages: Annotated[List[BaseMessage], manage_messages]
    
    # 任务队列：由 Planner 生成，调度中心根据此列表进行任务派发
    # 如果不定义此项，LangGraph 无法在节点间传递规划好的步骤
    plan: List[str] 
    
    # 历史执行结果：存放各专家节点（SQL/RAG）返回的原始数据
    # 使用 operator.add 确保多步任务的数据都能被顺序记录
    past_steps: Annotated[List[tuple], operator.add]
    
    # SQL 反思闭环专用变量
    current_sql: Optional[str]   # 暂存当前生成的 SQL 语句，供 Executor 执行
    sql_error: Optional[str]     # 记录执行报错，供 Generator 进行自我修正
    sql_retries: int             # 记录重试次数，配合路由逻辑防止死循环