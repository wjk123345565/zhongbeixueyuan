import sqlite3
import os

# 获取当前脚本所在目录的 admissions.db
db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "admissions.db"))

print("="*40)
print(f"🔍 正在扫描数据库文件：\n{db_path}")

if not os.path.exists(db_path):
    print("❌ 结果：文件根本不存在！请检查你的建库脚本把 db 存哪了。")
else:
    # 文件存在，连进去看看里面有什么表
    file_size = os.path.getsize(db_path)
    print(f"✅ 文件存在，文件大小: {file_size / 1024:.2f} KB")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # 查询 SQLite 的系统表，获取所有表名
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    if len(tables) == 0:
        print("❌ 结果：这是一个【空壳数据库】！里面没有任何表。")
    else:
        print(f"🎉 结果：找到表了！表名如下：")
        for t in tables:
            print(f"   - {t[0]}")
    conn.close()
print("="*40)