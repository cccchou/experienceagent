我需要解决从观察智能体OBSERVE agent传入的问题，我不清楚我的挖掘智能体如何接受上一个观察智能体传入的信息
在我的逻辑里：我传入观察智能体分配的信息，存入parsed_input，然后执行这个parsed_input。问题是如何和观察智能体进行连接？接受信息
`
def mine_best_workflow(self, user_input: str) -> str:
        """挖掘 Agent 根据用户输入调用观测 Agent 并通过 Aflow 框架搜索最佳工作流"""
        observation_agent = ObserveAgent()  # 调用观测 Agent
        parsed_input = observation_agent.observe(user_input)  # 获取任务需求

        best_workflow = self.mcts.search()  # 搜索最优工作流
        result = best_workflow.execute(parsed_input)
        return result
`