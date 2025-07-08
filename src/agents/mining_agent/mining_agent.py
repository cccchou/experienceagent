import random
import asyncio
from goalfy_core.runtime.local_runtime import LocalAgentRuntime
from goalfy_core.tps.tps_context import test_in_console
from prompt import generate_mining_dynamic_prompt 
from src.agents.launch_agent.launch_agent import new_launch_agent
from goalfy_core.llm_client import ToolCollection, LLM
from src.db import db_manager
from src.projects.config import app_config, simple_logging_setup
from src.projects.config import logger
#通过节点进行AFLOW操作，每个节点进行Prompt调用进行分析，寻找最佳的workflow，接受的信息来自launch_agent？observation_agent
#需要和launch_agent、observation_agent等进行协作，这部分的输入部分没有理解清楚，挖掘的逻辑根据论文已经完成了

# ====== 基本节点与工作流表示 ======

class ActionNode:
    def __init__(self, model, prompt, temperature=0, output_format="raw"):
        self.model = model
        self.prompt = prompt
        self.temperature = temperature
        self.output_format = output_format

    async def run(self, input_data):
        result = await self.model.call(self.prompt, input_data, temperature=self.temperature)
        return result

class Workflow:
    def __init__(self, name="intent_mining_workflow"):
        self.name = name
        self.nodes = []
        self.operators = []

    def add_node(self, node):
        self.nodes.append(node)

    def add_operator(self, operator):
        self.operators.append(operator)

    async def execute(self, input_data):
        results = []
        for node in self.nodes:
            res = await node.run(input_data)
            results.append(res)
        for op in self.operators:
            results = await op.apply(results, input_data)
        return results[-1] if results else None

# ====== 操作符模块 ======

class ReviewOperator:
    async def apply(self, solutions, problem):
        return [solutions[0]]

class SelfConsistencyOperator:
    async def apply(self, solutions, problem):
        return [max(set(solutions), key=solutions.count)]

class SelfRefineOperator:
    async def apply(self, solutions, problem):
        refined_solution = solutions[0] + " (Refined)"
        return [refined_solution]

# ====== 发起智能体 launch agent ======

class InitiatorAgent:
    def __init__(self, model):
        self.model = model

    async def initiate_task(self, user_input):
        # 发起智能体的任务发起
        task_content = await self.model.call("What is the user's intent?", user_input)
        print("[InitiatorAgent] Generated task:", task_content)
        return task_content

# ====== 挖掘智能体 ======

class IntentMiningAgent:
    def __init__(self, model, operators=None, max_iter=10):
        self.model = model
        self.max_iter = max_iter
        self.operators = operators if operators else []

    async def mine_intent(self, task_input, evaluator):
        best_workflow = None
        best_score = -float("inf")
        history = []
        prompt = generate_mining_dynamic_prompt(task_input)
        for i in range(self.max_iter):
            wf = self._generate_workflow(prompt=prompt)
            output = await wf.execute(task_input)
            score = evaluator(output)

            history.append((wf, score))
            if score > best_score:
                best_score = score
                best_workflow = wf

            print(f"[IntentMiningAgent] Iter {i+1}/{self.max_iter} | Score: {score}")

        return best_workflow, best_score, history

    def _generate_workflow(self):
        wf = Workflow()
        prompt = "What is your real need?"
        node = ActionNode(model=self.model, prompt=prompt, temperature=random.choice([0, 0.5, 1.0]))
        wf.add_node(node)

        if random.random() < 0.5 and self.operators:
            op = random.choice(self.operators)
            wf.add_operator(op)

        return wf

# ====== 示例 evaluator ======

def simple_evaluator(output):
    return len(str(output)) if output else 0

def new_mining_agent() -> IntentMiningAgent:
    """
    创建发起智能体实例的工厂函数
    
    Returns:
        配置完成的LaunchAgent实例
    """
    try:
        # 获取LLM配置
        llm = LLM(app_config.get_agent_llm_config("mining1"))
        
        # 创建智能体实例
        agent = IntentMiningAgent(model=llm, operators=[ReviewOperator(), SelfConsistencyOperator()])
        
        logger.info("挖掘智能体实例创建成功")
        return agent
        
    except Exception as e:
        logger.error(f"创建挖掘智能体失败: {e}", exc_info=True)
        raise e
# 集成示例：发起智能体与挖掘智能体的合作
if __name__=='__main__':
    LocalAgentRuntime.register_factory('挖掘智能体', new_mining_agent)
    # 设置日志级别
    simple_logging_setup()
    
    # 注册智能体工厂
    LocalAgentRuntime.register_factory('发起智能体', new_launch_agent)
    
    # # 测试运行
    # asyncio.run(test_in_console(
    #     db_manager,
    #     '发起智能体', 
    #     '我要创建一个品牌数据引擎系统人群创建流程的经验体', 
    #     session_id=22082, 
    #     session_metadata={'agent_type': '发起智能体'},
    #     once=True
    # ))

    async def run_integration():
        # 模型初始化
        initiator_model = LLM(app_config.get_agent_llm_config("consultation1"))  # 替换为实际的模型实例
        # 创建智能体
        initiator = InitiatorAgent(model=initiator_model)#发起智能体的内容
        miner = new_mining_agent()  # 挖掘智能体实例

        # 发起智能体任务
        user_input = "I need help with my project"
        task_content = await initiator.initiate_task(user_input)

        # 挖掘智能体进行意图挖掘
        best_wf, best_score, history = await miner.mine_intent(task_content, evaluator=simple_evaluator)

        # 使用最优工作流执行
        result = await best_wf.execute(task_content)
        print("Final Intent Extraction Result:", result)

    # 运行示例
    asyncio.run(run_integration())
