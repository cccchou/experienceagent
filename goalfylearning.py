import json
import os
import argparse
from openai import OpenAI
from experienceagent.fragment_recommender import ExperienceRetriever, FragmentRecommender
from experienceagent.fragment_scorer import ExperienceEvaluator
import logging
from experienceagent.controller_agent import ControllerAgent
from experienceagent.aflow_workflow import ConversationWorkflow

# 确保在环境变量中设置了 OPENAI_API_KEY
# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ExperienceSystem")


client = OpenAI(
    )

def call_openai(prompt: str, system_prompt: str = None, model: str = "deepseek-chat") -> str:
    """调用OpenAI API获取结果"""
    if system_prompt is None:
        system_prompt = "你是一个助理，将用户的回答总结成简练但信息完整的任务描述。"
        
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content
# 指导问题列表
QUESTIONS = [
    '请问你的目标是什么?',
    '你为什么需要这个功能?',
    '有哪些限制条件我们要考虑?',
    '你希望最终达到什么样的效果?'
]

def summarize_task(answers: list[str]) -> str:
    """
    使用 OpenAI Chat API 将用户的多条回答总结成一句详细的任务描述。
    """
    prompt = (
    "请将以下用户对话回答整合为一句详细的任务描述，用于启动智能体会话。"
    "仅输出一句话的纯文本，末尾不加句号，不要添加任何引号或其他符号，也不要输出多余的注释或解释：\n"
    + "\n".join(f"- {ans}" for ans in answers)
        )
    response = call_openai(prompt=prompt)
    return response.strip()


def run(db_path: str, use_dynamic_workflow: bool = True):
    db_path = "rich_expert_validation.json"
    # 初始化Evaluator、Retriever、Recommender
    evaluator = ExperienceEvaluator()
    retriever = ExperienceRetriever(db_path, evaluator)
    recommender = FragmentRecommender(retriever, evaluator)

    # 收集用户回答
    answers: list[str] = []
    task_description = ""

    print("\n====== GoalFy Learning Experience Agent ======")
    print("欢迎使用GoalFy学习体验智能体!")
    
    if use_dynamic_workflow:
        print("🚀 启动AFlow智能对话系统...")
        try:
            # 使用动态工作流
            workflow = ConversationWorkflow()
            workflow_result = workflow.run()
            
            task_description = workflow_result["task_description"]
            answers = workflow_result["collected_answers"]
            
            print(f"\n🎯 动态工作流生成的任务描述: {task_description}")
            
            # 显示对话统计信息
            context_state = workflow_result.get("context_state", {})
            print(f"📊 对话统计: {context_state.get('question_count', 0)} 个问题")
            print(f"🏷️ 识别领域: {context_state.get('domain_info', '通用')}")
            print(f"📋 任务类型: {context_state.get('task_type', '未知')}")
            
        except Exception as e:
            logger.error(f"动态工作流执行失败: {str(e)}, 切换到传统模式")
            use_dynamic_workflow = False
    
    if not use_dynamic_workflow:
        print("📋 使用传统固定问答模式...")
        print("告诉我您的需求，我会自动搜索并生成相关经验！")
        for i, q in enumerate(QUESTIONS, 1):
            print(f"\n[问题 {i}]")
            ans = input(q + " \n> ")
            answers.append(ans)

        # 总结为单句任务描述
        task_description = summarize_task(answers)
        print(f"\n📝 传统模式生成的任务描述: {task_description}")

    print(f"\n正在处理任务: {task_description}")

    # 初始化智能体并开始会话
    agent = ControllerAgent(db_path=db_path)
    session = agent.new_session(task_description)
    print(f"📋 会话开始: {session['message']}\n")

    # 运行智能体
    print("🔍 正在分析您的需求并搜索相关经验...")
    result = agent.run(answers)
    print(f"🎯 分析完成！\n")
    
    # 显示推荐结果
    items = result.get('WHY', [])
    print(f"💡 === 推荐结果 ===")
    if items:
        for i, item in enumerate(items, 1):
            goal = item['fragment']['data'].get('goal', '未知目标')
            similarity = item.get('similarity', 0)
            source = '🤖 AI生成' if item.get('source') == 'ai_generated' else '📚 经验库'
            reason = item.get("reason", "无")
            
            print(f"{i}. 任务: {goal}")
            print(f"   相似度: {similarity:.2f}")
            print(f"   来源: {source}")
            print(f'   原因: {reason}')
            print()
    else:
        print("暂未找到相关推荐")
    
    # 保存会话并写入文件
    save_result = agent.save_session(data=result)
    print(f"💾 保存会话结果: {save_result['message']}")
    agent.save_to_file()
    print("📁 已保存到 shuchu.json")
    result = agent.enhance_knowledge(force_rebuild=True)
    print(f"🧠 知识图谱增强结果: {result['message']}")

    # 结束提示
    print("✅ 所有操作已完成！")
    # Only ask for input if not in test mode
    try:
        input("\n按回车键退出程序。")
    except EOFError:
        # Handle case where input is mocked or not available
        pass
    
def main():
    """
    主函数，解析命令行参数并运行智能体。
    """
    parser = argparse.ArgumentParser(description="GoalFy Learning Experience Agent with AFlow Dynamic Workflow")
    parser.add_argument("--db_path", default="rich_expert_validation.json", help="经验库路径")
    parser.add_argument("--disable-dynamic", action="store_true", help="禁用动态工作流，使用传统固定问答")
    parser.add_argument("--mode", choices=["dynamic", "fixed"], default="dynamic", help="对话模式选择")
    args = parser.parse_args()
    
    # 确定使用的工作流模式
    use_dynamic_workflow = not args.disable_dynamic and args.mode == "dynamic"
    
    if use_dynamic_workflow:
        print("🚀 准备启动AFlow动态工作流模式")
    else:
        print("📋 准备启动传统固定问答模式")
    
    run(db_path=args.db_path, use_dynamic_workflow=use_dynamic_workflow)


if __name__ == "__main__":
    main()
