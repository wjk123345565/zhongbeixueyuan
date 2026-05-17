import sqlite3
import re
import os
from langchain_core.prompts import ChatPromptTemplate
from src.state import AgentState
from src.utils.llm import llm

# ==========================================
# 1. 辅助工具：动态获取数据库结构 (Schema)
# ==========================================
def get_db_schema(db_path: str) -> str:
    """
    改进：从数据库中动态读取表结构，避免硬编码，提高 Prompt 准确度
    """
    try:
        if not os.path.exists(db_path):
            return "未找到数据库文件"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # 获取 admission_scores 表的完整创建语句
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='admission_scores';")
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else "表 admission_scores 不存在"
    except Exception as e:
        return "获取 Schema 失败: " + str(e)

# ==========================================
# 2. SQL Generator (负责写 SQL 和修 Bug)
# ==========================================
sql_generation_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个资深的 SQLite 数据库专家，负责为中北学院招生系统生成查询语句。

【当前数据库表结构】：
{schema}

【极其重要的查询原则 (防查空兜底)】：
1. **安全第一**：严禁生成 DROP, DELETE, UPDATE, INSERT 等修改指令。只允许 SELECT。
2. **模糊匹配**：针对专业名称(major)、省份(province)、科类(category)，严禁使用等于号(=)，必须使用 `LIKE '%关键词%'`。
3. **💥核心词提取要求**：你绝对不能把用户的原话直接放进 LIKE 中！你必须对长句进行精简，只提取核心的 2-3 个字作为查询词根！
   - ✅ 正确：`province LIKE '%江苏%'` (去掉“省”、“市”)
   - ✅ 正确：`major LIKE '%计算机%'` (保留最核心专业词)
   - ✅ 正确：`category LIKE '%物理%'` (去掉“类”、“组”)
4. **分数比较法则**：如果用户询问“考了XX分能上什么专业”，必须使用 `min_score <= 考分`。
5. **输出规范**：只输出一行纯 SQL 语句，绝对不要包含 ```sql 等 Markdown 标记、解释说明或换行。

【执行反馈】：
如果下方存在“历史报错”，请务必分析错误原因并输出修正后的 SQL。"""),
    ("user", "【当前任务】：{current_task}\n【历史报错信息】：{sql_error}")
])

def sql_generator_node(state: AgentState):
    """
    SQL 生成节点：负责根据任务描述和错误反馈生成 SQL
    """
    current_task = state["plan"][0]
    error = state.get("sql_error", "")
    retries = state.get("sql_retries", 0)
    
    # 动态获取路径和 Schema 信息
    current_dir = os.path.dirname(__file__)
    db_path = os.path.abspath(os.path.join(current_dir, "../../admissions.db"))
    schema_info = get_db_schema(db_path)

    if error:
        print("🔄 [SQL Generator] 触发第 " + str(retries) + " 次自我修复！正在分析报错: " + str(error))
    else:
        print("🧠 [SQL Generator] 开始为任务编写 SQL: " + current_task)

    # 转义变量中的大括号，避免模板解析错误
    schema_info = schema_info.replace("{", "{{").replace("}", "}}")
    current_task = current_task.replace("{", "{{").replace("}", "}}")
    sql_error_msg = ("你上一次写的SQL报错了：" + error) if error else "无报错，请直接生成。"
    sql_error_msg = sql_error_msg.replace("{", "{{").replace("}", "}}")
    
    # 调用大模型生成 SQL
    chain = sql_generation_prompt | llm
    response = chain.invoke({
        "schema": schema_info,
        "current_task": current_task,
        "sql_error": sql_error_msg
    })
    
    # 清理 SQL 文本中的 Markdown 标记
    raw_sql = response.content.strip()
    clean_sql = re.sub(r"```sql|```|;", "", raw_sql).strip()
    
    # 安全拦截：防止非法 SQL 指令
    forbidden_words = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER"]
    if any(word in clean_sql.upper() for word in forbidden_words):
        print("⚠️ [SQL Generator] 检测到非法 SQL 指令，已拦截: " + clean_sql)
        return {"sql_error": "Security Breach: Unauthorized SQL Keywords Detected."}

    print("📝 [SQL Generator] 产出 SQL: " + clean_sql)
    
    # 将生成的 SQL 存入状态，供 Executor 使用
    return {"current_sql": clean_sql, "sql_retries": retries + 1}

# ==========================================
# 3. SQL Executor (负责执行并判定是否打回)
# ==========================================
def sql_executor_node(state: AgentState):
    """
    SQL 执行节点：负责数据库连接、结果映射及重试逻辑控制
    """
    sql = state.get("current_sql", "")
    current_task = state["plan"][0]
    retries = state.get("sql_retries", 0)
    max_retries = 3
    
    print("⚙️ [SQL Executor] 正在连接数据库执行验证...")
    
    try:
        # 动态定位数据库路径
        current_dir = os.path.dirname(__file__)
        db_path = os.path.abspath(os.path.join(current_dir, "../../admissions.db"))
        
        if not os.path.exists(db_path):
            raise FileNotFoundError("极其严重的路径错误：系统无法在 " + db_path + " 找到数据库文件！")
            
        conn = sqlite3.connect(db_path)
        # 改进：使用 Row 对象方便直接映射为带有列名的字典
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(sql)
        
        rows = cursor.fetchall()
        results = [dict(row) for row in rows]
        conn.close()
        
        # 逻辑改进：如果 SQL 运行成功但结果为空，视为“查空”错误，触发一次重试优化关键词
        if not results and retries < 2:
            print("🔍 [SQL Executor] SQL 运行成功但结果为空，尝试让生成器放宽查询条件...")
            return {"sql_error": "EMPTY_RESULT: 查询结果为空。请检查 LIKE 关键词是否太细，尝试提取更核心的词根。"}

        # 执行成功：核销任务并记录成果
        final_data = "执行 SQL: " + sql + " \n查询结果: " + str(results)
        print("✅ [SQL Executor] 执行成功！任务完成。数据: " + str(results))
        
        return {
            "past_steps": [(current_task, final_data)], 
            "plan": state["plan"][1:],                  # 划掉当前任务
            "sql_error": None,                          # 清空报错记录
            "sql_retries": 0,                           # 重置重试次数
            "current_sql": None
        }
        
    except Exception as e:
        error_msg = str(e)
        print("❌ [SQL Executor] 执行崩溃：" + error_msg)
        
        if retries >= max_retries:
            # 超过最大重试次数，宣告任务失败
            print("🛑 [SQL Executor] 重试次数耗尽，宣告任务失败。")
            final_data = "尝试 " + str(max_retries) + " 次后依然无法查询该数据。最后一次报错：" + error_msg
            return {
                "past_steps": [(current_task, final_data)],
                "plan": state["plan"][1:],
                "sql_error": None,
                "sql_retries": 0,
                "current_sql": None
            }
        else:
            # 将错误信息存入状态，由 graph.py 路由回 Generator
            return {"sql_error": error_msg}