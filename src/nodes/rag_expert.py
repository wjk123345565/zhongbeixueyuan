import json
import re
import os
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from src.state import AgentState

# ==========================================
# 1. 全局初始化：加载知识库 (只在系统导入时执行一次)
# ==========================================
print("⏳ [RAG Expert] 正在初始化本地政策知识库...")

embeddings = DashScopeEmbeddings(
    model="text-embedding-v2",
    dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY")
)

# 路径处理：确保能定位到项目根目录下的 JSON 知识库文件
current_dir = os.path.dirname(__file__)
json_path = os.path.abspath(os.path.join(current_dir, "../../data/unstructured/datasets-8CYn2dXuZbCO-alpaca-2026-05-15.json"))

docs = []
if os.path.exists(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
            
        for item in qa_data:
            question = item.get("instruction", "")
            raw_answer = item.get("output", "")
            # 改进：利用正则表达式剥离 DeepSeek 或其他模型的思维过程标签 (<think>...</think>)
            clean_answer = re.sub(r'<think>.*?</think>', '', raw_answer, flags=re.DOTALL).strip()
            
            if question and clean_answer:
                # 将问题存入 page_content 用于检索，答案存入 metadata 用于展示
                docs.append(Document(page_content=question, metadata={"answer": clean_answer}))
    except Exception as e:
        print(f"❌ [RAG Expert] 加载 JSON 失败: {str(e)}")
else:
    print(f"⚠️ [RAG Expert] 严重警告：未在 {json_path} 找到知识库文件！")

# 构建内存向量库
if docs:
    # 每次检索最相关的 2 条政策以保证上下文精简
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    print(f"✅ [RAG Expert] 知识库加载完成，共包含 {len(docs)} 条问答数据。")
else:
    retriever = None

# ==========================================
# 2. RAG 节点函数 (LangGraph 状态图调用入口)
# ==========================================
def rag_expert_node(state: AgentState):
    """
    RAG 专员节点：
    负责从本地知识库中提取与当前任务相关的政策信息。
    """
    # 检查任务清单是否为空
    if not state.get("plan"):
        return {"messages": state["messages"]}

    current_task = state["plan"][0]
    print(f"📚 [RAG Expert] 正在为任务检索知识库: {current_task}")

    if not retriever:
        final_data = "知识库未正确初始化或为空，无法检索相关政策。"
        print("❌ [RAG Expert] 检索失败：检索器不可用。")
    else:
        # 执行检索逻辑
        results = retriever.invoke(current_task)
        if not results:
            final_data = "在政策知识库中未检索到与该任务直接相关的信息。"
            print("🔍 [RAG Expert] 未命中任何相关政策。")
        else:
            # 组装格式化报告，提交给 synthesizer 汇总节点
            context_parts = []
            for i, res in enumerate(results):
                context_parts.append(
                    f"【相关资料 {i+1}】\n探讨问题：{res.page_content}\n原文规定：{res.metadata['answer']}"
                )
            
            final_data = "根据知识库检索得到以下权威信息：\n" + "\n\n".join(context_parts)

    print(f"✅ [RAG Expert] 检索任务完成。")

    # 更新状态：记录当前步骤成果，并从计划队列中移除已完成任务
    return {
        "past_steps": [(current_task, final_data)],
        "plan": state["plan"][1:] 
    }