import sqlite3
import csv
import os
import re

def init_db(db_path="admissions.db"):
    """初始化数据库并创建表（如果表已存在则不作处理）"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS admission_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            year INTEGER,
            province TEXT,
            category TEXT,
            subject_req TEXT,
            major TEXT,
            admit_count INTEGER,
            batch TEXT,
            max_score REAL,
            min_score REAL
        )
    ''')
    conn.commit()
    return conn

def extract_year_from_filename(filename):
    """使用正则表达式从文件名中提取连续的4位数字作为年份"""
    match = re.search(r'(20\d{2})', filename)
    if match:
        return int(match.group(1))
    return 2025 # 如果文件名没写年份，默认给个 2025 作为保底

def batch_import_scores(conn, data_folder="data/structured"):
    cursor = conn.cursor()
    total_files = 0
    total_rows = 0
    
    print(f"🔍 开始扫描目录: {data_folder} ...\n")
    
    # 遍历文件夹下的所有 txt 和 csv 文件
    for filename in os.listdir(data_folder):
        if not (filename.endswith('.txt') or filename.endswith('.csv')):
            continue
            
        file_path = os.path.join(data_folder, filename)
        year = extract_year_from_filename(filename)
        
        print(f"📄 正在解析: {filename} (自动识别年份: {year})")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None) # 跳过表头
            
            file_rows = 0
            for row in reader:
                if not row or len(row) < 8: continue
                try:
                    province, category, subject_req, major = row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip()
                    admit_count = int(row[4].strip()) if row[4].strip().isdigit() else 0
                    batch = row[5].strip()
                    max_score = float(row[6].strip()) if row[6].strip() else 0.0
                    min_score = float(row[7].strip()) if row[7].strip() else 0.0
                    
                    cursor.execute('''
                        INSERT INTO admission_scores 
                        (year, province, category, subject_req, major, admit_count, batch, max_score, min_score)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (year, province, category, subject_req, major, admit_count, batch, max_score, min_score))
                    
                    file_rows += 1
                except Exception as e:
                    pass # 静默忽略个别脏数据，保证大部队继续跑
            
            total_files += 1
            total_rows += file_rows
            print(f"   └── 成功导入 {file_rows} 条数据。")
            
    conn.commit()
    print(f"\n✅ 批量导入完成！共处理 {total_files} 个文件，向数据库新增 {total_rows} 条分数数据。")

if __name__ == "__main__":
    # 连接数据库并运行批量导入
    connection = init_db()
    batch_import_scores(connection)
    connection.close()