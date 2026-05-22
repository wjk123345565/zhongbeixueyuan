import json
import re
import os
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from src.state import AgentState

# ==========================================
# 1. 全局初始化：批量加载知识库 (只在系统导入时执行一次)
# ==========================================
print("⏳ [RAG Expert] 正在初始化本地政策知识库...")

# 使用 BGE 模型，支持精准的中文语义匹配
model_name = "BAAI/bge-small-zh-v1.5"
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name, 
    model_kwargs={'device': 'cpu'}, 
    encode_kwargs={'normalize_embeddings': True}
)

# 路径处理：确保能定位到项目根目录下的 unstructured 文件夹
current_dir = os.path.dirname(__file__)
data_dir = os.path.abspath(os.path.join(current_dir, "../../data/unstructured/"))

docs = []
loaded_files = 0

# 检查文件夹是否存在
if os.path.exists(data_dir):
    # 遍历该文件夹下的所有 .json 文件
    for filename in os.listdir(data_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(data_dir, filename)
            print(f"📄 [RAG Expert] 正在读取知识库文件: {filename}")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    qa_data = json.load(f)
                    
                for item in qa_data:
                    question = item.get("instruction", "")
                    raw_answer = item.get("output", "")
                    # 剥离 DeepSeek 或其他模型的思维过程标签 (<think>...</think>)
                    clean_answer = re.sub(r'<think>.*?</think>', '', raw_answer, flags=re.DOTALL).strip()
                    
                    if question and clean_answer:
                        # 将问题存入 page_content 用于检索，答案和文件来源存入 metadata 用于展示
                        docs.append(Document(
                            page_content=question, 
                            metadata={"answer": clean_answer, "source_file": filename}
                        ))
                loaded_files += 1
            except Exception as e:
                # 如果某个文件格式不对，只报警告，不影响其他文件加载
                print(f"❌ [RAG Expert] 加载 JSON 文件 {filename} 失败: {str(e)}")
else:
    print(f"⚠️ [RAG Expert] 严重警告：未找到知识库文件夹 {data_dir}！")

# 构建内存向量库
if docs:
    # 每次检索最相关的 2 条政策以保证上下文精简
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    print(f"✅ [RAG Expert] 知识库加载完成！共成功读取 {loaded_files} 个文件，包含 {len(docs)} 条问答数据。")
else:
    retriever = None

# ==========================================
# 2. RAG 节点函数 (LangGraph 状态图调用入口)
# ==========================================
def rag_expert_node(state: AgentState):
    """
    RAG 专员节点：
    负责从本地多文件中提取与当前任务相关的政策信息。
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
                # 增加溯源信息，标明来自哪个具体的 JSON 文件
                source = res.metadata.get("source_file", "未知文件")
                context_parts.append(
                    f"【相关资料 {i+1}】(来源文件: {source})\n探讨问题：{res.page_content}\n原文规定：{res.metadata['answer']}"
                )
            
            final_data = "根据知识库检索得到以下权威信息：\n" + "\n\n".join(context_parts)

    print(f"✅ [RAG Expert] 检索任务完成。")

    # 更新状态：记录当前步骤成果，并从计划队列中移除已完成任务
    return {
        "past_steps": [(current_task, final_data)],
        "plan": state["plan"][1:] 
    }