# mcts.py
import random
from workflow import Workflow

class MCTS:
    def __init__(self, iterations: int, workflows: List[Workflow], max_rounds: int = 100, early_stopping_rounds: int = 5):
        self.iterations = iterations
        self.workflows = workflows
        self.max_rounds = max_rounds
        self.early_stopping_rounds = early_stopping_rounds
        self.experiences = []  # 存储经验
        self.best_score = float('-inf')  # 初始化最佳评分

    def selection(self) -> Workflow:
        """选择操作：根据评估结果选择工作流"""
        workflow_scores = [self.evaluate(workflow) for workflow in self.workflows]
        max_score = max(workflow_scores)
        
        # 计算选择概率
        probabilities = [np.exp(score - max_score) for score in workflow_scores]
        probabilities /= sum(probabilities)  # 归一化
        selected_index = np.random.choice(range(len(self.workflows)), p=probabilities)
        return self.workflows[selected_index]

    def expansion(self, workflow: Workflow) -> Workflow:
        """扩展操作：生成新的工作流节点"""
        new_nodes = [Node(generate_node, "Step X", "Model-X")]  # 示例扩展
        expanded_workflow = Workflow(workflow.nodes + new_nodes)
        return expanded_workflow

    def evaluate(self, workflow: Workflow) -> float:
        """评估操作：执行工作流并返回评分"""
        input_data = "Initial input for testing workflow."
        output = workflow.execute(input_data)
        score = len(output)
        return score

    def backpropagation(self, selected_workflow: Workflow, score: float):
        """回溯操作：记录评估结果，更新父节点"""
        self.experiences.append((selected_workflow, score))

    def terminal_condition(self, current_round: int, best_score: float) -> bool:
        """终止条件：判断是否满足停止搜索的条件"""
        if current_round >= self.max_rounds:
            return True
        if best_score >= 0.95:
            return True
        return False

    def search(self) -> Workflow:
        """MCTS搜索过程"""
        best_workflow = None
        best_score = float('-inf')
        
        for round in range(self.iterations):
            workflow = self.selection()
            expanded_workflow = self.expansion(workflow)
            score = self.evaluate(expanded_workflow)
            
            # 记录评估结果
            self.backpropagation(expanded_workflow, score)
            
            if score > best_score:
                best_workflow = expanded_workflow
                best_score = score
            
            # 检查是否满足终止条件
            if self.terminal_condition(round, best_score):
                break

        return best_workflow
