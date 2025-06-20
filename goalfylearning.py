# goalfy_learning_framework.py
"""
通用 GoalFy Learning 框架
用于支持基于多模输入构建、演化与调用任务经验体（ExperiencePack）
适配场景：营销 / 客服 / 数据分析 / 产品搭建等
模块结构基于 6 个核心智能体阶段：
1. ControllerAgent - 阶段调度与流程判断
2. InitiationAgent - 任务意图判断与目标识别
3. ObservationAgent - 用户行为观察与操作结构化
4. ExtractionAgent - 客观知识点抽取（页面/控件/路径）
5. FusionAgent - 多源知识融合形成统一知识图谱
6. WorkflowBuilder - 构建任务执行工作流
"""

from typing import List, Dict, Any
from openai import OpenAI
from experienceagent.knowledge_graph import KnowledgePoint, ExperienceGraph
from experienceagent.agents.initiation_agent import InitiationAgent
from experienceagent.agents.observation_agent import ObservationAgent
from experienceagent.agents.extraction_agent import ExtractionAgent
from experienceagent.agents.fusion_agent import FusionAgent
from experienceagent.fragment_scorer import ExperienceEvaluator
from experienceagent.fragment_recommender import ExperienceRetriever
from experienceagent.controller_agent import ControllerAgent  # 添加控制器模块
import json

client = OpenAI(
        api_key = 'sk-8adcb7b1a1054215b485910737f07205',
        base_url='https://api.deepseek.com/v1'
        )

def call_openai(prompt: str) -> str:
    completion = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content

# -----------------------------
# Core Class Definitions
# -----------------------------
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

    def extract_from_fragments(self, types: List[str]) -> List[KnowledgePoint]:
        points = []
        for frag in self.fragments:
            if frag.frag_type in types:
                if frag.frag_type == "WHY":
                    data = frag.data
                    points.append(KnowledgePoint("GOAL", data.get("goal", ""), "subjective"))
                    points.append(KnowledgePoint("BACKGROUND", data.get("background", ""), "subjective"))
                    for i, c in enumerate(data.get("constraints", [])):
                        points.append(KnowledgePoint(f"CONSTRAINT_{i}", c, "subjective"))
                    points.append(KnowledgePoint("EXPECTED", data.get("expected_outcome", ""), "subjective"))
                elif frag.frag_type == "CHECK":
                    for i, rule in enumerate(frag.data.get("rules", [])):
                        points.append(KnowledgePoint(f"RULE_{i}", rule, "rule"))
        return points


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


class WorkflowBuilder:
    @staticmethod
    def build_from_fragments(fragments: List[ExperienceFragment]) -> Dict:
        workflow = {"steps": []}
        for frag in fragments:
            if frag.frag_type == "HOW":
                workflow["steps"].extend(frag.data.get("steps", []))
        return workflow


# -----------------------------
# Main Goalfy Learning Pipeline
# -----------------------------
if __name__ == "__main__":

    controller = ControllerAgent()
    # 模拟输入数据
    class MockInput:
        def __init__(self):
            self.interview_logs = [
                "我需要一个自动验证页面的工具",
                "用于检测页面变化并生成报告"
            ]
            self.behavior_logs = [
                {"action": "click", "element": "button", "time": 1623456789}
            ]
    
    # 执行流程
    result = controller.run_full_pipeline(MockInput())
    print(f"执行结果: {result['status']}")
    
    # 安全访问字典键
    final_stage = result.get('final_stage', '未知')
    print(f"最终阶段: {final_stage}")
    # controller.run_full_pipeline()
