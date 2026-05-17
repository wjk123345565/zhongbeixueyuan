from langchain_core.messages import HumanMessage
from src.graph import app

def run_chat_cli():
    """在终端运行的简易交互式测试循环"""
    print("="*50)
    print("欢迎使用中北学院招生多智能体系统测试版")
    print("输入 'q' 或 'quit' 退出")
    print("="*50)

    while True:
        user_input = input("\n[考生] 请输入你的问题: ").strip()
        
        if user_input.lower() in ['q', 'quit', 'exit']:
            print("退出系统。")
            break
            
        if not user_input:
            continue

        # 触发工作流
        result = app.invoke({"messages": [HumanMessage(content=user_input)]})
        
        # 提取最后一条消息作为助手的回复
        final_message = result['messages'][-1].content
        print("\n[招生助手]: " + final_message)

if __name__ == "__main__":
    run_chat_cli()