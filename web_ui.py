import os
from dotenv import load_dotenv
load_dotenv() # 必须在最前面加载环境变量

import time
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from src.graph import app # 导入编译好的 LangGraph 应用

# ==========================================
# 1. 页面配置与 UI 初始化
# ==========================================
st.set_page_config(
    page_title="中北学院招生智能体",
    page_icon="🎓",
    layout="centered"
)

# 自定义 CSS：美化聊天气泡与思考动画
st.markdown("""
<style>
    .stChatFloatingInputContainer { padding-bottom: 20px; }
    .thinking-text { color: #888888; font-size: 0.9em; font-style: italic; }
    
    /* 精准定位 Streamlit 原生的 user 头像标签，实现气泡靠右显示 */
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        flex-direction: row-reverse;
        text-align: right;
    }
    
    /* 美化用户的气泡，加入微信绿和圆润边角 */
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) div[data-testid="stMarkdownContainer"] {
        background-color: #95ec69; 
        color: black;
        padding: 10px 15px;
        border-radius: 15px 5px 15px 15px; 
        display: inline-block;
        text-align: left;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎓 中北学院招生智能体")
st.caption("基于 LangGraph 的多智能体协同系统 | 具备 RAG 政策检索与 SQL 精确查分能力")

# ==========================================
# 2. 会话状态管理 (维持对话上下文)
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="你好呀！我是中北学院专属招生助手。你可以问我历年分数线、转专业政策，或者专业介绍哦！")
    ]

# 渲染历史聊天记录
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# ==========================================
# 3. 处理用户输入与带动画的流式调用
# ==========================================
if prompt := st.chat_input("例如：2024年江苏物理类计算机专业最低分多少？"):
    
    # 1. 展示用户输入
    with st.chat_message("user"):
        st.markdown(prompt)
    
    st.session_state.messages.append(HumanMessage(content=prompt))
    
    # 2. 开启助手思考与执行流程
    with st.chat_message("assistant"):
        
        # 使用 st.status 创建可折叠的状态面板，展示 Agent 内部流转过程
        with st.status("🧠 智能体正在规划任务...", expanded=True) as status:
            final_response_content = ""
            
            try:
                # 核心逻辑：利用 app.stream() 实时捕捉节点切换事件
                for event in app.stream({"messages": st.session_state.messages}):
                    
                    # 遍历当前触发的节点名称
                    for node_name, node_state in event.items():
                        
                        # 针对不同节点类型展示特定动画日志
                        if node_name == "planner":
                            st.markdown("<span class='thinking-text'>📋 Planner 正在拆解您的咨询任务...</span>", unsafe_allow_html=True)
                        
                        elif node_name == "sql_generator":
                            st.markdown("<span class='thinking-text'>✍️ SQL Generator 正在编写精准查询语句...</span>", unsafe_allow_html=True)
                        
                        elif node_name == "sql_executor":
                            if node_state.get('sql_error'):
                                st.markdown("<span class='thinking-text'>⚠️ 执行遇到阻碍，准备自动修正 SQL...</span>", unsafe_allow_html=True)
                            else:
                                st.markdown("<span class='thinking-text'>📊 SQL Executor 成功从数据库提取录取数据...</span>", unsafe_allow_html=True)
                        
                        elif node_name == "rag_expert":
                            st.markdown("<span class='thinking-text'>📚 RAG Expert 正在从本地政策库检索条文...</span>", unsafe_allow_html=True)
                            
                        elif node_name == "chat_agent":
                            st.markdown("<span class='thinking-text'>💬 Chat Agent 正在组织回复...</span>", unsafe_allow_html=True)
                            final_response_content = node_state['messages'][-1].content
                            
                        elif node_name == "synthesizer":
                            st.markdown("<span class='thinking-text'>✨ Synthesizer 正在融会贯通，生成最终解答...</span>", unsafe_allow_html=True)
                            # 捕获最终汇总节点的输出
                            final_response_content = node_state['messages'][-1].content
                            
                        # 稍微延时让动画更清晰
                        time.sleep(0.3)
                
                # 流程结束，更新面板状态
                status.update(label="处理完成！", state="complete", expanded=False)
                
            except Exception as e:
                status.update(label="系统运行出错", state="error", expanded=True)
                st.error(f"Error: {str(e)}")
                st.stop()

        # 3. 模拟打字机特效输出最终回复
        message_placeholder = st.empty()
        full_response = ""
        # 逐字展示回复内容
        for chunk in list(final_response_content):
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")
            time.sleep(0.01) 
        
        message_placeholder.markdown(full_response)
        
        # 4. 更新对话历史
        st.session_state.messages.append(AIMessage(content=final_response_content))