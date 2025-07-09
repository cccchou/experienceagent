from typing import List, Callable

class Node:
    def __init__(self, operation: Callable, prompt: str, model: str):
        self.operation = operation
        self.prompt = prompt
        self.model = model

    def execute(self, input_data: str) -> str:
        """执行节点操作，返回结果"""
        return self.operation(input_data, self.prompt, self.model)

class Workflow:
    def __init__(self, nodes: List[Node]):
        self.nodes = nodes  # 工作流中的所有节点

    def execute(self, input_data: str) -> str:
        """执行整个工作流"""
        result = input_data
        for node in self.nodes:
            result = node.execute(result)  # 顺序执行所有节点
        return result