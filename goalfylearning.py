import json
import os
import argparse
from openai import OpenAI
from experienceagent.fragment_recommender import ExperienceRetriever, FragmentRecommender
from experienceagent.fragment_scorer import ExperienceEvaluator
import logging
from experienceagent.controller_agent import ControllerAgent

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


def run(db_path: str):
    db_path = "rich_expert_validation.json"
    # 初始化Evaluator、Retriever、Recommender
    evaluator = ExperienceEvaluator()
    retriever = ExperienceRetriever(db_path, evaluator)
    recommender = FragmentRecommender(retriever, evaluator)

    # 收集用户回答
    answers: list[str] = []

    print("\n====== GoalFy Learning Experience Agent ======")
    print("欢迎使用GoalFy学习体验智能体!")
    print("告诉我您的需求，我会自动搜索并生成相关经验！")
    for q in QUESTIONS:
        ans = input(q + " \n> ")
        answers.append(ans)
        # dialogue_history.append(ans)

    # 总结为单句任务描述
    task_description = summarize_task(answers)
    print(f"生成的任务描述: {task_description}\n")

    # 初始化智能体并开始会话
    agent = ControllerAgent(db_path=db_path)
    session = agent.new_session(task_description)
    print(f"会话开始: {session['message']}\n")

    # 运行智能体
    result = agent.run(answers)
    print(f"执行结果: {result}\n")
    items = result['WHY']
    print(f"=== 推荐 ===")
    items = result['WHY']
    for item in items:
        print(f"- 任务: {item['fragment']['data']['goal']}")
        print(f"  相似度: {item['similarity']:.2f}")
        print(f"  来源: {'AI生成' if item.get('source') == 'ai_generated' else '经验库'}")
        print(f'  原因: {item.get("reason", "无")}')
    # 保存会话并写入文件
    save_result = agent.save_session(data=result)
    print(f"保存会话结果: {save_result['message']}")
    agent.save_to_file()
    print("已保存到 shuchu.json")
    result = agent.enhance_knowledge(force_rebuild=True)
    print(f"知识图谱增强结果: {result['message']}")

    # 结束提示
    input("\n所有操作已完成，按回车键退出程序。")
    
def main():
    """
    主函数，解析命令行参数并运行智能体。
    """
    parser = argparse.ArgumentParser(description="GoalFy Learning Experience Agent")
    parser.add_argument("--db_path", default="rich_expert_validation.json", help="经验库路径")
    args = parser.parse_args()
    
    run(db_path=args.db_path)


if __name__ == "__main__":
    main()
