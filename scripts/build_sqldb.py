import sqlite3
import csv
import os

def create_database(db_path="admissions.db"):
    """初始化 SQLite 数据库并创建分数表"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 创建一个高度结构化的数据表
    # 注意：我们这里手动加上了 year（年份）字段
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS admission_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            year INTEGER,          -- 年份 (我们从代码里补上)
            province TEXT,         -- 省份
            category TEXT,         -- 科类 (如艺术类、普通类)
            subject_req TEXT,      -- 选科要求
            major TEXT,            -- 专业
            admit_count INTEGER,   -- 录取人数
            batch TEXT,            -- 批次
            max_score REAL,        -- 最高分
            min_score REAL         -- 最低分
        )
    ''')
    conn.commit()
    return conn

def import_txt_to_db(conn, file_path, year):
    """读取纯文本/CSV数据并插入数据库"""
    cursor = conn.cursor()
    
    print(f"正在处理: {file_path} (年份设定为: {year})")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # 既然数据是用逗号分隔的，我们直接用 csv 模块读取
        reader = csv.reader(f)
        
        # 跳过第一行的表头
        next(reader, None)
        
        insert_count = 0
        for row in reader:
            # 过滤掉空行
            if not row or len(row) < 8:
                continue
                
            # 提取我们需要的前 8 列数据 (忽略平信志愿、征求志愿等可能为空的列)
            # row 的对应关系: 0:省份, 1:科类, 2:选科, 3:专业, 4:人数, 5:批次, 6:最高分, 7:最低分
            try:
                province = row[0].strip()
                category = row[1].strip()
                subject_req = row[2].strip()
                major = row[3].strip()
                admit_count = int(row[4].strip()) if row[4].strip().isdigit() else 0
                batch = row[5].strip()
                max_score = float(row[6].strip()) if row[6].strip() else 0.0
                min_score = float(row[7].strip()) if row[7].strip() else 0.0
                
                # 插入数据库
                cursor.execute('''
                    INSERT INTO admission_scores 
                    (year, province, category, subject_req, major, admit_count, batch, max_score, min_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (year, province, category, subject_req, major, admit_count, batch, max_score, min_score))
                
                insert_count += 1
            except Exception as e:
                print(f"解析行数据出错: {row}，错误信息: {e}")
                
    conn.commit()
    print(f"✅ 成功导入 {insert_count} 条数据！\n")

if __name__ == "__main__":
    # 确保我们在项目根目录运行，生成的 db 文件放在根目录
    db_path = "admissions.db"
    conn = create_database(db_path)
    
    # ==========================================
    # ⚠️ 请在这里修改你的本地 TXT 文件路径和对应的年份
    # ==========================================
    # 假设你把 TXT 文件放在了 data/ 目录下
    txt_file_path = "data/jiangsu_scores.txt"  # 替换为你真实的文件名
    
    # 确保文件存在再执行
    if os.path.exists(txt_file_path):
        # 传入连接、文件路径，以及这批数据对应的年份
        import_txt_to_db(conn, txt_file_path, year=2025)
    else:
        print(f"❌ 找不到文件: {txt_file_path}，请检查路径是否正确。")
        
    conn.close()
    print("🎉 数据库构建完毕！")