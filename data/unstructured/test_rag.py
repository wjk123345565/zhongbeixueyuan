import json
import re
from langchain_core.documents import Document
from langchain_chroma import Chroma
# 这里我们用最轻量免费的 HuggingFace 词向量模型做测试
from langchain_community.embeddings import HuggingFaceBgeEmbeddings 

def build_and_test_rag(json_path):
    print("1. 正在读取并清洗问答对数据...")
    with open(json_path, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
        
    docs = []
    for item in qa_data:
        question = item.get("instruction", "")
        raw_answer = item.get("output", "")
        
        # 💥 核心清洗魔法：用正则表达式把 <think>...</think> 及其内部的所有内容删掉
        clean_answer = re.sub(r'<think>.*?</think>', '', raw_answer, flags=re.DOTALL).strip()
        
        # 如果问题和答案都不为空，则制作成诱饵文档
        if question and clean_answer:
            # page_content 放问题（用于向量检索），metadata 放干净的答案（用于最终回复）
            doc = Document(
                page_content=question, 
                metadata={"answer": clean_answer}
            )
            docs.append(doc)
            
    print(f"✅ 成功清洗并构建了 {len(docs)} 个问答诱饵。")

    print("\n2. 正在加载词向量模型并构建本地 Chroma 数据库 (初次运行可能需要下载模型)...")
    # 使用 BGE 模型，这是目前开源界对中文支持最好的轻量级向量模型之一
    model_name = "BAAI/bge-small-zh-v1.5"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    # 构建内存级向量库（测试用）
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) # 每次找最相关的2个问题
    print("✅ 向量数据库构建完毕！\n")

    # ================= 测试环节 =================
    test_queries = [
        "新闻学这个专业主要是培养什么人的？核心思想是啥？", # 口语化测试
        "你们中北学院到底是谁办的？迁址到哪里了？"           # 混合提问测试
    ]

    for query in test_queries:
        print("="*50)
        print(f"🧑‍🎓 模拟学生提问: {query}")
        results = retriever.invoke(query)
        
        print("\n🤖 检索到的知识片段 (将发给汇总员):")
        for i, res in enumerate(results):
            # 打印命中的原问题和对应的干净答案
            print(f"\n--- 片段 {i+1} ---")
            print(f"🎯 命中了系统问题: {res.page_content}")
            print(f"📖 对应的纯净答案: {res.metadata['answer']}")

if __name__ == "__main__":
    # 填入你刚才上传的 JSON 文件的实际路径
    build_and_test_rag("D:/PythonProjects/NnuZcAgent/data/unstructured/datasets-8CYn2dXuZbCO-alpaca-2026-05-15.json")