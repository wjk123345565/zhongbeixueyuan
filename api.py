# api.py
import os
from dotenv import load_dotenv

# 🚀 必须在导入我们自己的模块之前，先把 .env 里的钥匙加载进系统环境！
load_dotenv() 

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List

# 现在再导入 agent_app，它底层去初始化 LLM 时就能找到 Key 了
from src.graph import app as agent_app

app = FastAPI(title="中北学院招生 Agent API")

# 允许跨域请求（让前端能访问到）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 生产环境请改为前端的真实域名
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

@app.post("/api/chat")
async def chat_stream(request: ChatRequest):
    """接收前端的历史记录，返回流式响应"""
    # 转换前端的格式为 LangChain 的格式
    formatted_messages = [
        {"role": m.role, "content": m.content} for m in request.messages
    ]
    
    async def generate_stream():
        # 调用 LangGraph 的异步流式输出
        async for event in agent_app.astream_events({"messages": formatted_messages}, version="v1"):
            kind = event["event"]
            # 捕获最终大模型的文本块并实时推给前端
            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk.content:
                    yield chunk.content
                    
    return StreamingResponse(generate_stream(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)