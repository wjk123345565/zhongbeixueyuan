from langchain_core.messages import SystemMessage, HumanMessage
from src.state import AgentState
from src.utils.llm import llm

def chat_agent_node(state: AgentState):
    """接待智能体：处理日常寒暄"""
    query = state["messages"][-1].content
    
    system_prompt = """你是中北学院的招生小助手，请用热情、亲切的语气回答用户的日常问候或常规闲聊。

    【输出格式强制警告】：
    绝对、永远、严禁在回复中使用波浪号 (~) 作为语气词或标点符号！
    因为这会触发前端 Markdown 的删除线渲染 Bug。请全部使用句号 (。)、感叹号 (!) 或 Emoji 表情来表达语气。
    """
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ])
    
    print(f"[Chat Agent] 处理日常闲聊。")
    return {"messages": state["messages"] + [response]}