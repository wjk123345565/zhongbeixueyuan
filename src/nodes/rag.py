# import os
# from langchain_core.messages import SystemMessage, HumanMessage
# from langchain_community.embeddings import DashScopeEmbeddings
# from langchain_community.vectorstores import Chroma
# from src.state import AgentState
# from src.utils.llm import llm

# # ==========================================
# # 安全获取 API Key (兼容本地和云端)
# # ==========================================
# api_key = os.environ.get("DASHSCOPE_API_KEY")
# if not api_key:
#     try:
#         import streamlit as st
#         api_key = st.secrets.get("DASHSCOPE_API_KEY")
#     except Exception:
#         pass

# # 全局初始化一次向量数据库连接，显式传入 api_key
# embeddings = DashScopeEmbeddings(
#     model="text-embedding-v2",
#     dashscope_api_key=api_key
# )

# vector_db = Chroma(
#     persist_directory="./chroma_db", 
#     embedding_function=embeddings
# )

# # ...下面的 def rag_agent_node(state: AgentState): 保持完全不变...
# def rag_agent_node(state: AgentState):
#     """升级版政策解答智能体：真实连接 ChromaDB 并带调试打印"""
#     query = state["messages"][-1].content
    
#     print(f"[RAG Agent] 正在 ChromaDB 中检索...")
    
#     # 1. 向量相似度检索，取最相关的 3 个片段
#     docs = vector_db.similarity_search(query, k=3)
    
#     # 🌟 增加透视日志：打印到底搜出了什么
#     print(f"\n=== [Debug] 查找到 {len(docs)} 条相关片段 ===")
#     for i, doc in enumerate(docs):
#         print(f"片段 {i+1}: {doc.page_content}")
#     print("=============================================\n")
    
#     # 2. 将检索到的文档内容拼接成上下文
#     retrieved_context = "\n".join([doc.page_content for doc in docs])
    
#     # 如果没查到任何东西，给一个保底的上下文提示
#     if not retrieved_context.strip():
#         retrieved_context = "未能在中北学院知识库中检索到相关信息。"
    
#     # 3. 构建包含检索结果的 Prompt
#     system_prompt = (
#         "你是中北学院的官方招生咨询助手。请严格根据以下【已知信息】回答用户问题。\n"
#         "如果已知信息无法回答该问题，请直接说明“抱歉，目前的招生资料中未包含该信息”，切勿凭空捏造。\n\n"
#         f"【已知信息】:\n{retrieved_context}"
#     )
    
#     # 4. 调用大模型生成最终回复
#     response = llm.invoke([
#         SystemMessage(content=system_prompt),
#         HumanMessage(content=query)
#     ])
    
#     print(f"[RAG Agent] 基于检索内容生成最终回复。")
#     # 修改后 (引入 Plan-and-Execute 的状态更新)：
#     def rag_agent_node(state: AgentState):
#         # 1. 看看自己被派了什么活
#         current_task = state["plan"][0] 
    
#         # ... 中间去 ChromaDB 检索和生成回答的代码保持不变 ...
#         # 假设最终检索生成的答案存在 result_text 变量中
    
#         # 2. 向上级交差：上报成果，划掉任务！
#         return {
#             "past_steps": [(current_task, result_text)], # 记录：(我干了啥, 成果是啥)
#             "plan": state["plan"][1:] # 切片魔法：把第0个任务剔除，剩下的任务还给 state
#     }

import os
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from src.state import AgentState
from src.utils.llm import llm

# ==========================================
# 1. 初始化向量库 (保持之前的云端兼容逻辑)
# ==========================================
api_key = os.environ.get("DASHSCOPE_API_KEY")
if not api_key:
    try:
        import streamlit as st
        api_key = st.secrets.get("DASHSCOPE_API_KEY")
    except Exception:
        api_key = "sk-empty-key"

embeddings = DashScopeEmbeddings(model="text-embedding-v2", dashscope_api_key=api_key)
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# ==========================================
# 2. 设计 RAG 专员的 Prompt
# ==========================================
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的【中北学院】招生政策专家。
你的任务是根据提供的背景资料，精准回答子任务中的政策问题。

【要求】：
- 仅根据背景资料回答，不要编造。
- 保持回答简洁、客观。
- 如果资料中没有提到，请如实说明。"""),
    ("user", "【背景资料】：\n{context}\n\n【具体子任务】：{current_task}")
])

def rag_agent_node(state: AgentState):
    """
    RAG 政策专员：根据子任务检索知识库并上报结果。
    """
    # 1. 认领任务
    current_task = state["plan"][0]
    print(f"\n📚 [RAG Agent] 开始执行子任务 ➡️ {current_task}")

    # 2. 检索向量数据库
    docs = vector_db.similarity_search(current_task, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    # 3. 生成回答
    chain = rag_prompt | llm
    response = chain.invoke({"context": context, "current_task": current_task})
    result_text = response.content
    
    print(f"✅ [RAG Agent] 政策检索完毕。")

    # 4. 核销任务并上报成果
    return {
        "past_steps": [(current_task, result_text)],
        "plan": state["plan"][1:] 
    }