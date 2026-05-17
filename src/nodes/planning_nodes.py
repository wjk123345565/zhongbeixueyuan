import os
import json
from typing import List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from src.state_planning import PlanningState
from src.utils.llm import llm

def search_policy_node(state: PlanningState) -> dict:
    query = state["messages"][-1].content
    system_prompt = """你是一个学习规划政策研究员。你的任务是根据用户的需求，从本地知识库中检索相关的学习政策、考试大纲、复习方法指导等信息。

    【你的职责】：
    1. 理解用户的学科背景和目标
    2. 搜索相关的政策文件和复习指导
    3. 整理检索到的关键信息

    【输出格式】：
    请输出你检索到的政策要点摘要，格式如下：
    - 相关政策1: [简要描述]
    - 相关政策2: [简要描述]
    """

    embeddings = DashScopeEmbeddings(
        model="text-embedding-v2",
        dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY")
    )

    json_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/unstructured/datasets-8CYn2dXuZbCO-alpaca-2026-05-15.json"))
    docs = []
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
            for item in qa_data:
                question = item.get("instruction", "") or item.get("question", "")
                answer = item.get("output", "") or item.get("answer", "")
                if question and answer:
                    docs.append(Document(page_content=question, metadata={"answer": answer}))

    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    results = retriever.invoke(query)

    context = "\n".join(["【相关资料】" + r.page_content + "\n" + r.metadata['answer'] for r in results])

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "用户需求：" + query + "\n\n检索到的资料：\n" + context)
    ])

    response = llm.invoke(prompt.invoke({}))
    return {"messages": [AIMessage(content="📚 【政策检索结果】\n\n" + response.content)], "current_plan": response.content}


def gather_intel_node(state: PlanningState) -> dict:
    query = state["messages"][-1].content
    system_prompt = """你是一个学习情报分析师。你的任务是根据用户的情况，收集整理相关的学习情报：

    【你需要收集的信息】：
    1. 学科特点分析
    2. 知识点难度评估
    3. 时间分配建议
    4. 资源需求分析

    【工作原则】：
    - 基于用户提供的信息进行分析
    - 给出具体可操作的情报建议
    - 标注重点和难点

    【输出格式】：
    请以结构化方式输出你的分析结果。
    """

    # 转义查询中的大括号，避免模板解析错误
    query = query.replace("{", "{{").replace("}", "}}")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "请分析以下学习需求：" + query)
    ])

    response = llm.invoke(prompt.invoke({}))
    return {"messages": [AIMessage(content="🔍 【情报收集结果】\n\n" + response.content)]}


def drafter_node(state: PlanningState) -> dict:
    system_prompt = """你是一个高考复习规划专家。你的任务是为即将参加【高考】的学生制定复习计划。

    【重要声明】：
    - 这是【高考复习计划】，不是考研复习！
    - 高考科目包括：语文、数学、英语、理科综合（物理、化学、生物）或文科综合（政治、历史、地理）
    - 高考数学内容为高中数学，包含函数、导数基础、不等式、数列、概率统计等，【不包含】高等数学、线性代数
    - 高考物理包含力学、电磁学、光学、热学等高中物理，【不包含】大学物理、电路分析、电子学
    - 高考化学包含有机化学、无机化学、化学反应原理等高中化学，【不包含】大学有机化学

    【起草原则】：
    1. 计划必须具体、可执行、有时间节点
    2. 要平衡各学科的学习时间
    3. 要考虑学生的心理状态和激励因素
    4. 要标注重点复习章节
    5. 根据用户提供的文理科意向，合理安排科目

    【计划结构要求】：
    ## 学习目标
    [具体可衡量的目标，如：总分提升XX分，考上XX大学XX专业]

    ## 时间规划
    - 基础阶段（第X-X周）：夯实基础，梳理知识点
    - 强化阶段（第X-X周）：专题突破，提升解题能力
    - 冲刺阶段（第X-X周）：模拟训练，查漏补缺

    ## 学科复习安排（根据考生类型调整）

    ### 语文
    [复习内容，如：现代文阅读、文言文阅读、作文写作等]

    ### 数学（文科/理科）
    [文科/理科数学复习内容]

    ### 英语
    [复习内容，如：词汇、语法、阅读理解、写作等]

    ### 综合科目（文科/理科）
    [根据文理科选择：
    - 理科：物理+化学+生物
    - 文科：政治+历史+地理]

    ## 重点关注
    - 薄弱知识点1
    - 薄弱知识点2

    ## 心理调适建议
    [如何保持良好心态，合理安排休息]
    """

    messages = state["messages"]
    context = "\n".join([msg.content for msg in messages if isinstance(msg, AIMessage)])
    
    # 转义上下文中的大括号，避免模板解析错误
    context = context.replace("{", "{{").replace("}", "}}")

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "请基于以下信息起草学习计划：\n\n" + context)
    ])

    response = llm.invoke(prompt.invoke({}))
    return {"current_plan": response.content, "iteration_count": 0}


def reviewer_academic_node(state: PlanningState) -> dict:
    plan = state.get("current_plan", "")
    
    # 转义计划中的大括号，避免模板解析错误
    plan = plan.replace("{", "{{").replace("}", "}}")
    
    system_prompt = """你是一个学术审查员，负责评估学习计划的科学性和可行性。

    【你的职责】：
    - 评估学习计划的知识覆盖完整性和时间安排合理性
    - 如果计划基本合理，应该给出通过意见

    【审查标准】：
    只要计划满足以下基本条件，就应该通过：
    1. 有明确的学习目标
    2. 时间安排基本合理
    3. 包含具体的学习内容
    4. 难度梯度适中

    【输出格式】：
    如果计划基本合格，请输出：[合格] + 简要肯定评价
    如果计划有严重问题必须修改，请输出：[需修改] + 具体修改建议

    注意：不要过于严格，只要计划基本合理就应该通过。
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "请审查以下学习计划：\n\n" + plan)
    ])

    response = llm.invoke(prompt.invoke({}))
    academic_vote = response.content

    return {"academic_votes": [academic_vote]}


def reviewer_emotional_node(state: PlanningState) -> dict:
    plan = state.get("current_plan", "")
    
    # 转义计划中的大括号，避免模板解析错误
    plan = plan.replace("{", "{{").replace("}", "}}")
    
    system_prompt = """你是一个心理审查员，负责评估学习计划的心理负担和激励性。

    【你的职责】：
    - 评估学习计划是否会让学生压力过大
    - 判断计划是否具有激励性，能否维持学习动力
    - 如果计划心理负担适中，就应该给出通过意见

    【审查标准】：
    只要计划满足以下基本条件，就应该通过：
    1. 每天学习时间不超过合理上限（建议6-8小时）
    2. 有适当的休息和放松时间
    3. 目标设定积极但可达成
    4. 包含正向激励机制

    【输出格式】：
    如果计划心理负担适中，请输出：[合格] + 简要肯定评价
    如果计划压力过大必须调整，请输出：[需修改] + 具体调整建议

    注意：不要过于严格，只要计划心理负担可接受就应该通过。
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "请审查以下学习计划的心理负担：\n\n" + plan)
    ])

    response = llm.invoke(prompt.invoke({}))
    emotional_vote = response.content

    return {"emotional_votes": [emotional_vote]}


def consensus_check_node(state: PlanningState) -> dict:
    academic_votes = state.get("academic_votes", [])
    emotional_votes = state.get("emotional_votes", [])

    latest_academic = academic_votes[-1] if academic_votes else ""
    latest_emotional = emotional_votes[-1] if emotional_votes else ""

    academic_pass = "[合格]" in latest_academic
    emotional_pass = "[合格]" in latest_emotional

    if academic_pass and emotional_pass:
        return {
            "consensus_reached": True,
            "needs_revision": False,
            "revision_feedback": None
        }
    else:
        feedback_parts = []
        if not academic_pass:
            feedback_parts.append("【学术审查意见】：" + latest_academic)
        if not emotional_pass:
            feedback_parts.append("【心理审查意见】：" + latest_emotional)

        return {
            "consensus_reached": False,
            "needs_revision": True,
            "revision_feedback": "\n\n".join(feedback_parts)
        }


def plan_output_node(state: PlanningState) -> dict:
    plan = state.get("current_plan", "")
    return {"final_plan": plan}


def revision_node(state: PlanningState) -> dict:
    plan = state.get("current_plan", "")
    feedback = state.get("revision_feedback", "")
    iteration = state.get("iteration_count", 0)

    if iteration >= 3:
        return {
            "final_plan": plan,
            "needs_revision": False
        }

    # 转义变量中的大括号，避免模板解析错误
    plan = plan.replace("{", "{{").replace("}", "}}")
    feedback = feedback.replace("{", "{{").replace("}", "}}")

    system_prompt = """你是一个学习计划修订专家。用户对当前的学习计划提出了修改意见，你需要据此进行修订。

    【修订原则】：
    1. 认真听取反馈意见
    2. 保持计划的核心结构
    3. 只修改被指出问题的部分
    4. 修订后保持计划的可执行性

    【输出要求】：
    请输出修订后的完整学习计划，标注【修订内容】的部分。
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "原始计划：\n" + plan + "\n\n修改意见：\n" + feedback)
    ])

    response = llm.invoke(prompt.invoke({}))
    return {
        "current_plan": response.content,
        "iteration_count": iteration + 1,
        "academic_votes": [],
        "emotional_votes": []
    }