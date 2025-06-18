# controller_agent.py
"""
总控智能体 ControllerAgent
用于管理各阶段 Agent 的执行顺序、状态迁移与异常回退逻辑
"""

from typing import Dict, Literal

Stage = Literal[
    "initiation", "observation", "extraction", "fusion", "practice"
]

class ControllerAgent:
    def __init__(self):
        self.current_stage: Stage = "initiation"
        self.stage_history: list[Stage] = ["initiation"]

    def update_context(self, context: Dict[str, bool]) -> Stage:
        """
        根据上下文判断下一阶段
        context 包含阶段评估结果，例如：
        {
            "confirmed": True,
            "valuable": True,
            "logs_complete": True,
            "within_scope": True,
            "subjective_complete": False,
            "fusion_success": False,
            "errors": False
        }
        """
        next_stage = self.current_stage

        if self.current_stage == "initiation":
            if context.get("confirmed") and context.get("valuable"):
                next_stage = "observation"

        elif self.current_stage == "observation":
            if context.get("logs_complete") and context.get("within_scope"):
                next_stage = "extraction"
            elif not context.get("logs_complete"):
                next_stage = "initiation"

        elif self.current_stage == "extraction":
            if context.get("subjective_complete"):
                next_stage = "fusion"
            elif context.get("missing_observation"):
                next_stage = "observation"

        elif self.current_stage == "fusion":
            if context.get("fusion_success"):
                next_stage = "practice"

        elif self.current_stage == "practice":
            if context.get("errors"):
                next_stage = "extraction"

        # 更新历史和当前状态
        if next_stage != self.current_stage:
            self.stage_history.append(next_stage)
            self.current_stage = next_stage

        return self.current_stage

    def get_state(self):
        return {
            "current": self.current_stage,
            "history": self.stage_history
        }

# 示例用法
def example():
    controller = ControllerAgent()
    ctx = {
        "confirmed": True,
        "valuable": True,
        "logs_complete": True,
        "within_scope": True,
        "subjective_complete": True,
        "fusion_success": True,
        "errors": False
    }
    for _ in range(5):
        next_stage = controller.update_context(ctx)
        print("Current stage:", next_stage)

if __name__ == "__main__":
    example()
