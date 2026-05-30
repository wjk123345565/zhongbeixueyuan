import os
from dotenv import load_dotenv
load_dotenv()

import time
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from src.graph import app as chat_app
from src.planning_graph import planning_app

# ====================== 这里添加了自动隐藏侧边栏 ======================
st.set_page_config(
    page_title="中北学院招生智能体",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"  # 自动隐藏侧边栏（手机/电脑都生效）
)
# ======================================================================

st.markdown("""
<style>
    .stChatFloatingInputContainer { padding-bottom: 20px; }
    .thinking-text { color: #888888; font-size: 0.9em; font-style: italic; }

    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        flex-direction: row-reverse;
        text-align: right;
    }

    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) div[data-testid="stMarkdownContainer"] {
        background-color: #95ec69;
        color: black;
        padding: 10px 15px;
        border-radius: 15px 5px 15px 15px;
        display: inline-block;
        text-align: left;
    }

    .sidebar-section {
        padding: 10px;
        margin: 5px 0;
        border-radius: 8px;
        background-color: #f0f2f6;
    }

    .plan-display {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }

    .workflow-step {
        padding: 8px 12px;
        margin: 5px 0;
        border-radius: 5px;
        background-color: #e7f5ff;
        border-left: 4px solid #339af0;
    }

    .vote-pass {
        background-color: #d3f9d8;
        border-left-color: #51cf66;
        padding: 10px;
        border-radius: 5px;
    }

    .vote-fail {
        background-color: #fff3bf;
        border-left-color: #fcc419;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

if "current_page" not in st.session_state:
    st.session_state.current_page = "对话"

if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="你好呀！我是中北学院专属招生助手。你可以问我历年分数线、转专业政策，或者专业介绍哦！")
    ]

if "planning_messages" not in st.session_state:
    st.session_state.planning_messages = []

if "final_plan" not in st.session_state:
    st.session_state.final_plan = None

if "plan_awaiting_approval" not in st.session_state:
    st.session_state.plan_awaiting_approval = False

if "plan_draft" not in st.session_state:
    st.session_state.plan_draft = None

st.markdown("""
<style>
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
        padding-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 📋 功能菜单")

    if st.button("💬 对话", use_container_width=True, type="primary" if st.session_state.current_page == "对话" else "secondary"):
        st.session_state.current_page = "对话"
        st.rerun()

    if st.button("📚 复习规划", use_container_width=True, type="primary" if st.session_state.current_page == "复习规划" else "secondary"):
        st.session_state.current_page = "复习规划"
        st.rerun()

    st.markdown("---")
    st.markdown("### ℹ️ 关于")
    st.markdown("中北学院招生智能体 v1.0")

def render_chat_page():
    st.title("💬 招生咨询")
    st.caption("基于 LangGraph 的多智能体协同系统 | 具备 RAG 政策检索与 SQL 精确查分能力")

    for msg in st.session_state.messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    if prompt := st.chat_input("例如：2024年江苏物理类计算机专业最低分多少？"):
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.messages.append(HumanMessage(content=prompt))

        with st.chat_message("assistant"):
            with st.status("🧠 智能体正在规划任务...", expanded=True) as status:
                final_response_content = ""

                try:
                    for event in chat_app.stream({"messages": st.session_state.messages}):
                        for node_name, node_state in event.items():
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
                                final_response_content = node_state['messages'][-1].content

                            time.sleep(0.3)

                    status.update(label="处理完成！", state="complete", expanded=False)

                except Exception as e:
                    status.update(label="系统运行出错", state="error", expanded=True)
                    st.error("Error: " + str(e))
                    st.stop()

            message_placeholder = st.empty()
            full_response = ""
            for chunk in list(final_response_content):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.01)

            message_placeholder.markdown(full_response)

            st.session_state.messages.append(AIMessage(content=final_response_content))

def render_planning_page():
    st.title("📚 学习规划（对抗式）")
    st.markdown("### 🎯 多智能体协同学习计划生成系统")

    st.markdown("""
    <div style="background-color: #e7f5ff; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
    <h4>🔄 工作流程说明</h4>
    <ol>
        <li><b>起草者 (Drafter)</b> - 根据需求生成学习计划</li>
        <li><b>学术审查员</b> - 评估计划的科学性和可行性</li>
        <li><b>情绪审查员</b> - 评估计划的心理负担和激励性</li>
        <li><b>共识检查</b> - 双票通过才能输出，否则打回重写</li>
        <li><b>人工介入 (HIL)</b> - 您可以审核、编辑或要求重写计划</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col2:
        st.markdown("### ⚙️ 当前状态")
        if st.session_state.plan_awaiting_approval:
            st.info("📝 等待您的审核...")
        elif st.session_state.final_plan:
            st.success("✅ 计划已生成")
        else:
            st.info("📋 等待开始规划")

        if st.session_state.final_plan:
            st.download_button(
                label="📥 下载计划",
                data=st.session_state.final_plan,
                file_name="学习计划.md",
                mime="text/markdown"
            )

    with col1:
        user_input = st.text_area(
            "请描述您的学习需求：",
            placeholder="例如：我是2025届高考考生，理科，目标分数600分，想报考计算机专业，请帮我制定复习计划",
            height=120,
            key="planning_input"
        )

        col_start, col_clear = st.columns(2)

        with col_start:
            if st.button("🚀 开始规划", type="primary", use_container_width=True):
                if user_input.strip():
                    st.session_state.planning_messages = [HumanMessage(content=user_input)]
                    st.session_state.plan_awaiting_approval = False
                    st.session_state.final_plan = None
                    st.session_state.plan_draft = None

                    with st.status("🔄 对抗式规划进行中...", expanded=True) as status:
                        plan_content = ""
                        current_draft = ""
                        academic_votes = []
                        emotional_votes = []
                        iteration = 0

                        try:
                            for event in planning_app.stream({"messages": st.session_state.planning_messages}):
                                for node_name, node_state in event.items():
                                    if node_name == "search_policy":
                                        st.markdown("<span class='workflow-step'>📚 政策检索员正在搜索相关政策...</span>", unsafe_allow_html=True)
                                        time.sleep(0.5)

                                    elif node_name == "gather_intel":
                                        st.markdown("<span class='workflow-step'>🔍 情报分析师正在收集学习情报...</span>", unsafe_allow_html=True)
                                        time.sleep(0.5)

                                    elif node_name == "drafter":
                                        st.markdown("<span class='workflow-step'>✍️ 起草专家正在制定学习计划...</span>", unsafe_allow_html=True)
                                        if node_state.get("current_plan"):
                                            current_draft = node_state["current_plan"]
                                        time.sleep(0.5)

                                    elif node_name == "reviewer_academic":
                                        st.markdown("<span class='workflow-step'>🎓 学术审查员正在审查...</span>", unsafe_allow_html=True)
                                        if node_state.get("academic_votes"):
                                            vote = node_state["academic_votes"][-1]
                                            academic_votes.append(vote)
                                            time.sleep(0.5)

                                    elif node_name == "reviewer_emotional":
                                        st.markdown("<span class='workflow-step'>💚 情绪审查员正在审查...</span>", unsafe_allow_html=True)
                                        if node_state.get("emotional_votes"):
                                            vote = node_state["emotional_votes"][-1]
                                            emotional_votes.append(vote)
                                            time.sleep(0.5)

                                    elif node_name == "consensus_check":
                                        st.markdown("<span class='workflow-step'>✅ 共识检查中...</span>", unsafe_allow_html=True)
                                        if node_state.get("academic_votes") and node_state.get("emotional_votes"):
                                            latest_academic = node_state["academic_votes"][-1]
                                            latest_emotional = node_state["emotional_votes"][-1]

                                            st.markdown("<div class='vote-pass'>🎓 学术审查：" + latest_academic + "</div>", unsafe_allow_html=True)
                                            st.markdown("<div class='vote-pass'>💚 情绪审查：" + latest_emotional + "</div>", unsafe_allow_html=True)

                                        if node_state.get("needs_revision"):
                                            st.markdown("<span class='workflow-step'>🔄 需要修订，打回重写...</span>", unsafe_allow_html=True)
                                            iteration += 1
                                            st.markdown("<div class='vote-fail'>⚠️ 第 " + str(iteration) + " 轮修订</div>", unsafe_allow_html=True)
                                        time.sleep(0.5)

                                    elif node_name == "plan_output":
                                        if node_state.get("final_plan"):
                                            plan_content = node_state["final_plan"]
                                        time.sleep(0.5)

                            status.update(label="对抗式规划完成！", state="complete", expanded=False)

                        except Exception as e:
                            status.update(label="规划出错", state="error", expanded=True)
                            st.error("错误: " + str(e))
                            st.stop()

                        if current_draft:
                            st.session_state.plan_draft = current_draft

                        if academic_votes and emotional_votes:
                            latest_academic = academic_votes[-1]
                            latest_emotional = emotional_votes[-1]
                            academic_pass = "[合格]" in latest_academic
                            emotional_pass = "[合格]" in latest_emotional

                            if academic_pass and emotional_pass:
                                st.session_state.plan_awaiting_approval = True
                            else:
                                st.session_state.plan_awaiting_approval = True
                        else:
                            st.session_state.plan_awaiting_approval = True

                    st.rerun()
                else:
                    st.warning("请输入您的学习需求")

        with col_clear:
            if st.button("🗑️ 清空", use_container_width=True):
                st.session_state.planning_messages = []
                st.session_state.plan_awaiting_approval = False
                st.session_state.final_plan = None
                st.session_state.plan_draft = None
                st.rerun()

    st.markdown("---")

    if st.session_state.plan_draft:
        st.markdown("### 📝 当前计划草稿")
        plan_display = st.session_state.plan_draft.replace("{", "&#123;").replace("}", "&#125;")
        st.markdown("<div class='plan-display'>" + plan_display + "</div>", unsafe_allow_html=True)

        if st.session_state.plan_awaiting_approval and not st.session_state.final_plan:
            st.markdown("### 👤 人工介入 (Human-in-the-Loop)")
            st.markdown("请审核以上学习计划，选择您的操作：")

            col_approve, col_revise_minor, col_revise_full = st.columns(3)

            with col_approve:
                if st.button("✅ 确认通过", type="primary", use_container_width=True):
                    st.session_state.final_plan = st.session_state.plan_draft
                    st.session_state.plan_awaiting_approval = False
                    st.success("计划已确认！")
                    st.rerun()

            with col_revise_minor:
                minor_feedback = st.text_input("微调意见：", placeholder="请输入需要微调的内容...")
                if st.button("🔧 轻微修改", use_container_width=True, disabled=not minor_feedback):
                    st.session_state.plan_awaiting_approval = False
                    st.warning("功能开发中：请稍后重试")

            with col_revise_full:
                if st.button("🔄 完全重写", type="secondary", use_container_width=True):
                    st.session_state.plan_draft = None
                    st.session_state.plan_awaiting_approval = False
                    st.info("已发起重写请求，请重新点击「开始规划」")
                    st.rerun()

            with st.expander("✏️ 直接编辑计划"):
                edited_plan = st.text_area(
                    "您可以直接编辑下方的学习计划：",
                    value=st.session_state.plan_draft,
                    height=400,
                    key="edited_plan"
                )

                if st.button("💾 保存编辑", type="primary"):
                    st.session_state.final_plan = edited_plan
                    st.session_state.plan_awaiting_approval = False
                    st.success("计划已保存！")
                    st.rerun()

    if st.session_state.final_plan:
        st.markdown("### 🎉 最终学习计划")
        final_display = st.session_state.final_plan.replace("{", "&#123;").replace("}", "&#125;")
        st.markdown("<div class='plan-display'>" + final_display + "</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    if st.session_state.current_page == "对话":
        render_chat_page()
    elif st.session_state.current_page == "复习规划":
        render_planning_page()