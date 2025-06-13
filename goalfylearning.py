# goalfy_learning_framework.py
"""
通用 GoalFy Learning 框架
用于支持基于多模输入构建、演化与调用任务经验体（ExperiencePack）
适配场景：营销 / 客服 / 数据分析 / 产品搭建等
模块结构基于 4 个核心阶段：
1. 多模输入采集
2. 知识建构与经验沉淀
3. 经验体演化与质量更新
4. 面向任务的经验激活与工作流生成
"""

from typing import List, Dict, Any
from openai import OpenAI
from experienceagent.knowledge_graph import KnowledgePoint, ExperienceGraph
import json

client = OpenAI()

def call_openai(prompt: str) -> str:
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content

class MultiModalInput:
    def __init__(self):
        self.interview_logs: List[str] = []
        self.behavior_logs: List[Dict[str, Any]] = []
        self.uploaded_assets: List[str] = []
        self.context_knowledge: Dict[str, Any] = {}

    def ingest_interview(self, text: str):
        self.interview_logs.append(text)

    def record_behavior(self, log: Dict[str, Any]):
        self.behavior_logs.append(log)

    def upload_asset(self, path: str):
        self.uploaded_assets.append(path)

    def update_context(self, key: str, value: Any):
        self.context_knowledge[key] = value


class ExperienceFragment:
    def __init__(self, frag_type: str):
        self.frag_type = frag_type
        self.data = {}


class ExperiencePack:
    def __init__(self, task_name: str):
        self.task_name = task_name
        self.fragments: List[ExperienceFragment] = []
        self.trust_score = 0.5
        self.version = 1
        self.kg = ExperienceGraph(task_name)

    def add_fragment(self, frag: ExperienceFragment):
        self.fragments.append(frag)

    def summarize(self):
        return {
            'task': self.task_name,
            'version': self.version,
            'trust': self.trust_score,
            'fragments': [f.frag_type for f in self.fragments]
        }

    def extract_knowledge_from_fragments(self):
        for frag in self.fragments:
            if frag.frag_type == "WHY":
                data = frag.data
                self.kg.add_kp(KnowledgePoint("GOAL", data.get("goal", ""), "subjective"))
                self.kg.add_kp(KnowledgePoint("BACKGROUND", data.get("background", ""), "subjective"))
                for i, c in enumerate(data.get("constraints", [])):
                    self.kg.add_kp(KnowledgePoint(f"CONSTRAINT_{i}", c, "subjective"))
                self.kg.add_kp(KnowledgePoint("EXPECTED", data.get("expected_outcome", ""), "subjective"))
            elif frag.frag_type == "HOW":
                steps = frag.data.get("steps", [])
                for i, step in enumerate(steps):
                    kp = KnowledgePoint(f"STEP_{i}", f"在页面{step['page']}执行{step['action']}于{step['element']}", "objective", url=step['page'])
                    self.kg.add_kp(kp)


class WhyBuilder:
    @staticmethod
    def from_interview(logs: List[str]) -> ExperienceFragment:
        prompt = f"""
你是一个经验建构助手。请阅读以下用户访谈内容，并抽取其中的 4 类信息：
1. 目标（Goal）
2. 背景（Background）
3. 约束条件（Constraints）
4. 预期效果（Expected Outcome）
格式：
{{
  "goal": "...",
  "background": "...",
  "constraints": ["...", "..."],
  "expected_outcome": "..."
}}
访谈内容：
{chr(10).join(logs)}
"""
        response = call_openai(prompt)
        try:
            structured = json.loads(response)
        except:
            structured = {
                "goal": logs[0],
                "background": logs[1] if len(logs) > 1 else "",
                "constraints": logs[2:] if len(logs) > 2 else [],
                "expected_outcome": "Not specified"
            }
        frag = ExperienceFragment("WHY")
        frag.data = structured
        return frag


class HowBuilder:
    @staticmethod
    def from_behavior(steps: List[Dict]) -> ExperienceFragment:
        prompt = f"""
你是一个用户行为结构化助手，请将以下用户行为记录整理为结构化步骤：
- 页面名称（page）
- 动作类型（action）
- 操作元素（element）
- 意图说明（intent）
行为记录：
{steps}
返回 JSON 数组
"""
        try:
            response = call_openai(prompt)
            structured = json.loads(response)
        except:
            structured = steps

        frag = ExperienceFragment("HOW")
        frag.data = {"steps": structured}
        return frag


class CheckBuilder:
    @staticmethod
    def from_rules(rules: List[str]) -> ExperienceFragment:
        prompt = f"""
你是流程规则生成器，请根据任务规则生成校验逻辑，格式：JSON 数组。
任务信息：{rules}
"""
        try:
            response = call_openai(prompt)
            structured = json.loads(response)
        except:
            structured = rules

        frag = ExperienceFragment("CHECK")
        frag.data = {"rules": structured}
        return frag


class ExperienceEvaluator:
    @staticmethod
    def update_trust(ep: ExperiencePack, success: bool, feedback_score: float):
        delta = 0.1 if success else -0.1
        ep.trust_score = max(0, min(1.0, ep.trust_score + delta * feedback_score))
        return ep.trust_score

    @staticmethod
    def evolve(ep: ExperiencePack, feedback_summary: str):
        prompt = f"以下是用户反馈，请改进原有经验体：\n{feedback_summary}\n经验体摘要：{ep.summarize()}"
        updated_info = call_openai(prompt)
        frag = ExperienceFragment("WHY")
        frag.data = {"evolution": updated_info}
        ep.version += 1
        ep.add_fragment(frag)
        return ep


class ExperienceRetriever:
    def __init__(self, all_experiences: List[ExperiencePack]):
        self.repo = all_experiences

    def match_by_task(self, task_query: str) -> List[ExperiencePack]:
        return [ep for ep in self.repo if task_query in ep.task_name]


class WorkflowBuilder:
    @staticmethod
    def build_from_fragments(fragments: List[ExperienceFragment]) -> Dict:
        workflow = {"steps": []}
        for frag in fragments:
            if frag.frag_type == "HOW":
                workflow["steps"].extend(frag.data.get("steps", []))
        return workflow


if __name__ == "__main__":
    input_module = MultiModalInput()
    input_module.ingest_interview("我要做一个618广告活动")
    input_module.ingest_interview("因为活动多且时间紧，想自动化")
    input_module.ingest_interview("预算不能超过500元")
    input_module.record_behavior({"page": "campaign", "action": "click", "element": "#create-btn", "timestamp": "T1"})

    ep = ExperiencePack("营销活动流程自动化")
    ep.add_fragment(WhyBuilder.from_interview(input_module.interview_logs))
    ep.add_fragment(HowBuilder.from_behavior(input_module.behavior_logs))
    ep.add_fragment(CheckBuilder.from_rules(["budget <= 500"]))
    ep.extract_knowledge_from_fragments()
    ep.kg.visualize_console()

    ExperienceEvaluator.update_trust(ep, success=True, feedback_score=1.0)
    ExperienceEvaluator.evolve(ep, feedback_summary="流程中有多余步骤，建议简化")

    matched = ExperienceRetriever([ep]).match_by_task("营销")
    workflow = WorkflowBuilder.build_from_fragments(matched[0].fragments)
    print("[Workflow Execution Plan]", workflow)
