from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from src.state import AgentState
from src.utils.llm import llm

# ==========================================
# 1. 综合汇总专员的 Prompt 设计
# ==========================================
synthesizer_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个热情、专业的【中北学院】招生办老师。
你的任务是：根据其他后台专员为你提供的【查证事实】，来回答用户的【原始问题】。

【工作原则】：
1. 必须且只能基于【查证事实】作答，严禁脑补或编造任何分数、位次和政策！
2. 如果事实中包含报错信息，请委婉地向同学说明该部分数据暂时无法查询，但要输出已查到的其他有效信息。
3. 语气要亲切、自然，展现出大学老师的耐心。可以使用适当的 emoji 增加亲和力。
4. 注意排版的美观性，关键的分数、专业名、核心条件请使用 **粗体** 标注，条理清晰。
"""),
    ("user", "【同学的原始问题】：\n{question}\n\n【后台专员提供的查证事实】：\n{facts}")
])

def synthesizer_node(state: AgentState):
    """
    综合汇总节点：将所有任务的执行结果揉成一段完美的人类语言。
    """
    print("\n✨ [Synthesizer] 调度中心已确认任务全部完成，正在撰写最终报告...")
    
    # 1. 提取用户的原始问题 (通常是消息列表里的最后一条用户的发言)
    original_question = state["messages"][-1].content
    
    # 2. 组装所有执行专员交上来的“证据”
    past_steps = state.get("past_steps", [])
    facts = ""
    for i, (task_name, task_result) in enumerate(past_steps, 1):
        facts += f"--- 事实 {i} ---\n【执行任务】：{task_name}\n【查证结果】：{task_result}\n\n"
        
    print(f"📦 [Synthesizer] 收集到的证据池：\n{facts}")
    
    # 3. 唤醒大模型进行最终的润色和总结
    chain = synthesizer_prompt | llm
    response = chain.invoke({
        "question": original_question,
        "facts": facts
    })
    
    final_answer = response.content
    print(f"🎉 [Synthesizer] 最终回复撰写完毕！")
    
    # 4. 完美谢幕：
    # - 将最终回答打包成 AIMessage 追加到聊天记录中供前端展示
    # - (可选) 可以选择在这里清空 past_steps，不过通常前端新起一次对话时会重置 state
    return {
        "messages": [AIMessage(content=final_answer)]
    }