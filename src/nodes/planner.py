from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from src.state import AgentState
from src.utils.llm import llm

# ==========================================
# 1. 强制约束输出结构 (利用 Pydantic)
# ==========================================
class Plan(BaseModel):
    """计划任务列表约束"""
    steps: List[str] = Field(
        description="为了回答用户问题，按顺序拆解出的各个子任务步骤列表。任务描述需包含所有必要实体（如年份、省份、专业）。"
    )

# ==========================================
# 2. 设计主管的 Prompt
# ==========================================
system_prompt = """你是一个专门为【中北学院】招生办设计的“任务规划主管（Planner）”。
你的目标是将用户复杂的提问，精准拆解为一系列可被下游专员独立执行的子任务。

你目前手下有三个专员可以调遣：
1. 【SQL数据查询专员】：仅负责查询历年各省市、各专业的录取分数线、位次、招生计划等结构化数字。
2. 【RAG政策检索专员】：仅负责查询学校的招生简章、转专业政策、宿舍情况、奖学金等纯文字条款。
3. 【闲聊接待专员】：负责处理“你好”、“你是谁”、“感谢”等无具体业务诉求的日常问候。

【拆解规则】：
- 如果问题仅涉及查分，只需生成1个【SQL数据查询】任务。
- 如果问题仅涉及政策，只需生成1个【RAG政策检索】任务。
- 💥 如果问题极其复杂，必须拆解为多个独立任务！
- 任务描述必须非常具体，把用户问题中的省份、文理科、专业等关键信息带入任务中，绝对不能出现代词。


【经典拆解案例】：
用户：我是江苏理科生，想报计算机，往年分数线多少？进去后转软件工程政策是什么？
你的思考：这个问题既需要查分，又需要查政策。
你的拆解结果应该类似：
[
    "调用SQL专员：查询江苏省理科计算机专业的历年录取分数线", 
    "调用RAG专员：查询中北学院理科生转专业到软件工程的相关政策条件"
]

注意：用户通常会使用极简的口语短句（如“江苏理科，软工，多少分”）。
只要用户的意图是在询问分数、专业、政策或录取情况，你都必须进行任务拆解！绝对不能把它当作闲聊！
"""

planner_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "用户的当前提问是：{question}")
])

# ==========================================
# 3. 组装 Chain (魔法在这里：with_structured_output)
# ==========================================
# 这行代码强制要求千问大模型必须且只能返回符合 Plan 类的结构数据
planner_chain = planner_prompt | llm.with_structured_output(Plan)

# ==========================================
# 4. 定义 Graph 节点函数
# ==========================================
def planner_node(state: AgentState):
    # 拿到用户最新的一句话
    question = state['messages'][-1].content
    
    print("🧠 Planner 正在思考如何拆解任务：" + question)
    
    try:
        # 唤醒大模型进行推理拆解
        plan_result = planner_chain.invoke({"question": question})
        
        # 🛡️ 防御性编程 1：如果大模型返回了 None（通常发生在面对纯闲聊时）
        if plan_result is None or not hasattr(plan_result, 'steps'):
            print("📋 Planner 发现这是简单对话，无需复杂拆解。")
            return {"plan": ["调用闲聊专员：回复用户的日常对话"]}
            
        # 🛡️ 防御性编程 2：如果大模型返回了空列表 []
        if len(plan_result.steps) == 0:
            print("📋 Planner 生成了空任务单，自动转入闲聊兜底。")
            return {"plan": ["调用闲聊专员：回复用户的日常对话"]}
            
        print("📋 Planner 拆解出的任务清单：" + str(plan_result.steps))
        return {"plan": plan_result.steps}
        
    except Exception as e:
        # 🛡️ 终极兜底：万一大模型网络波动或抛出任何异常，系统不崩溃，直接让闲聊专员安抚用户
        print("⚠️ Planner 解析发生异常: " + str(e))
        return {"plan": ["调用闲聊专员：安抚用户，说明系统正在开小差"]}