import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

def build_batch_vector_db(data_folder="data/unstructured"):
    print(f"1. 正在扫描目录 {data_folder} 中的所有纯文本文件...")
    
    # 使用 DirectoryLoader 批量加载文件夹内所有的 .txt 文件
    # 如果你以后有 pdf，可以引入 PyPDFLoader 并修改 glob="**/*.pdf"
    loader = DirectoryLoader(
        data_folder, 
        glob="**/*.txt", 
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'}
    )
    docs = loader.load()
    
    if not docs:
        print("❌ 未在目录中找到任何 txt 文件，请检查路径。")
        return

    print(f"   └── 成功加载了 {len(docs)} 个文档文件。")

    print("2. 正在进行批量文本切分...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    print(f"   └── 已将所有文档切分为 {len(splits)} 个知识片段。")

    print("3. 正在调用千问模型进行向量化并存入 ChromaDB...")
    embeddings = DashScopeEmbeddings(model="text-embedding-v2")
    
    # 使用 from_documents 重新生成数据库
    Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db" 
    )
    
    print("✅ 知识库批量构建完成！数据已更新至 ./chroma_db。")

if __name__ == "__main__":
    build_batch_vector_db()