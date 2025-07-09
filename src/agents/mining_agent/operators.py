# operators.py

from abc import ABC, abstractmethod
from prompt import (
    DEFAULT_GENERATE_PROMPT,
    DEFAULT_REVIEW_PROMPT,
    DEFAULT_REVISE_PROMPT,
    DEFAULT_FORMAT_PROMPT,
    DEFAULT_ENSEMBLE_PROMPT
)

class Operator(ABC):
    def __init__(self, llm):
        self.llm = llm  # 语言模型实例

    @abstractmethod
    async def __call__(self, *args, **kwargs):
        pass


class Generate(Operator):
    async def __call__(self, problem: str) -> str:
        """
        使用生成专家的提示，根据用户问题生成详细回答。
        """
        prompt = DEFAULT_GENERATE_PROMPT.format(problem=problem)
        return await self.llm.generate(prompt)


class Review(Operator):
    async def __call__(self, problem: str, solution: str) -> str:
        """
        审阅专家对已有回答进行评估，输出反馈意见。
        """
        prompt = DEFAULT_REVIEW_PROMPT.format(problem=problem, solution=solution)
        return await self.llm.generate(prompt)


class Revise(Operator):
    async def __call__(self, problem: str, solution: str, feedback: str) -> str:
        """
        修订专家根据审阅反馈优化答案。
        """
        prompt = DEFAULT_REVISE_PROMPT.format(problem=problem, solution=solution, feedback=feedback)
        return await self.llm.generate(prompt)


class Format(Operator):
    async def __call__(self, problem: str, solution: str, type:str) -> str:
        """
        格式化专家将答案整理成清晰结构。
        """
        prompt = DEFAULT_FORMAT_PROMPT.format(problem=problem, solution=solution, type=type)
        return await self.llm.generate(prompt)


class Ensemble(Operator):
    async def __call__(self, problem: str, solutions: list[str]) -> str:
        """
        融合专家整合多个解答为一个更优答案。
        """
        formatted_solutions = "\n".join([f"- {s}" for s in solutions])
        prompt = DEFAULT_ENSEMBLE_PROMPT.format(problem=problem, solutions=formatted_solutions)
        return await self.llm.generate(prompt)
