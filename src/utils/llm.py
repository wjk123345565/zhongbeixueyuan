import os
from langchain_community.chat_models import ChatTongyi
from dotenv import load_dotenv

load_dotenv()
# 尝试获取 API Key
api_key = os.environ.get("DASHSCOPE_API_KEY")

# 如果在 Streamlit 环境下且普通环境变量没读到，尝试直接从 st.secrets 读取
if not api_key:
    try:
        import streamlit as st
        api_key = st.secrets.get("DASHSCOPE_API_KEY")
    except Exception:
        pass

# 显式传入 api_key 参数，防止 Pydantic 验证报错
llm = ChatTongyi(
    model="qwen-turbo",
    dashscope_api_key=api_key
)