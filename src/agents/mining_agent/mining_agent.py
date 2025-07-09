# mining_agent.py
from typing import List
from goalfy_core.llm_client.client import LLM
from prompt import GENERATE_PROMPT, FORMAT_PROMPT
from mcts import MCTS
from workflow import Workflow, Node
from operators import Generate, Format, Review, Revise, Ensemble
# from llm import LLM  # 假设我们有一个 LLM 类来进行调用
from observe_agent import ObserveAgent  # 导入已实现的观测 Agent

# 挖掘 Agent
class MiningAgent:
    def __init__(self, workflows: List[Workflow], iterations: int):
        self.mcts = MCTS(iterations, workflows)

    def mine_best_workflow(self, user_input: str) -> str:
        """挖掘 Agent 根据用户输入调用观测 Agent 并通过 Aflow 框架搜索最佳工作流"""
        observation_agent = ObserveAgent()  # 调用观测 Agent
        parsed_input = observation_agent.observe(user_input)  # 获取任务需求

        best_workflow = self.mcts.search()  # 搜索最优工作流
        result = best_workflow.execute(parsed_input)
        return result

if __name__ == "__main__":
    generate_node = Node(Generate(LLM()), "Step 1: Generate", "Model-A")
    workflow1 = Workflow([generate_node])  # 示例工作流

    # 创建挖掘 Agent 实例
    mining_agent = MiningAgent(workflows=[workflow1], iterations=100)

    # 用户输入需求
    user_input = "Solve the problem: What is the square root of 16?"#这一部分是来自观察智能体输出内容

    # 挖掘 Agent 搜索并执行最佳工作流
    result = mining_agent.mine_best_workflow(user_input)
    print(f"Final result: {result}")