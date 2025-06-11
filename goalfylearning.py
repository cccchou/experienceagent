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

# -----------------------------
# Stage 1: 多模输入理解与用户数据采集
# -----------------------------
from typing import List, Dict, Any
from openai import OpenAI

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
        self.interview_logs: List[str] = []  # 访谈语句
        self.behavior_logs: List[Dict[str, Any]] = []  # 操作行为 [{page, action, element, timestamp}]
        self.uploaded_assets: List[str] = []  # 上传文件路径 / 图像等
        self.context_knowledge: Dict[str, Any] = {}  # 上下文知识图谱

    def ingest_interview(self, text: str):
        self.interview_logs.append(text)

    def record_behavior(self, log: Dict[str, Any]):
        self.behavior_logs.append(log)

    def upload_asset(self, path: str):
        self.uploaded_assets.append(path)

    def update_context(self, key: str, value: Any):
        self.context_knowledge[key] = value


# -----------------------------
# Stage 2: 知识建构与经验沉淀
# -----------------------------
class ExperienceFragment:
    def __init__(self, frag_type: str):
        self.frag_type = frag_type  # 'WHY' | 'HOW' | 'CHECK'
        self.data = {}

class ExperiencePack:
    def __init__(self, task_name: str):
        self.task_name = task_name
        self.fragments: List[ExperienceFragment] = []
        self.trust_score = 0.5
        self.version = 1

    def add_fragment(self, frag: ExperienceFragment):
        self.fragments.append(frag)

    def summarize(self):
        return {
            'task': self.task_name,
            'version': self.version,
            'trust': self.trust_score,
            'fragments': [f.frag_type for f in self.fragments]
        }

# 构建 WHY 片段
class WhyBuilder:
    @staticmethod
    def from_interview(logs: List[str]) -> ExperienceFragment:
        prompt = f"""
你是一个经验建构助手。请阅读以下用户访谈内容，并抽取其中的 4 类信息：
1. 目标（Goal） - 用户想达成什么？
2. 背景（Background） - 用户为什么要做这件事？
3. 约束条件（Constraints） - 实现中有哪些限制？
4. 预期效果（Expected Outcome） - 希望最终获得的效果是什么？
请用如下 JSON 格式返回：
{{
  "goal": "...",
  "background": "...",
  "constraints": ["...", "..."],
  "expected_outcome": "..."
}}
访谈内容如下：
{chr(10).join(logs)}
"""
        response = call_openai(prompt)
        try:
            import json
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

# 构建 HOW 片段
class HowBuilder:
    @staticmethod
    def from_behavior(steps: List[Dict]) -> ExperienceFragment:
        prompt = f"""
你是一个用户行为结构化助手，请根据以下用户行为记录将其整理为执行步骤列表，每一步包含：
- 页面名称（page）
- 动作类型（action）
- 操作元素（element）
- 意图说明（推测该行为的目的）
行为记录：
{steps}
返回 JSON 数组
"""
        try:
            import json
            response = call_openai(prompt)
            structured = json.loads(response)
        except:
            structured = steps

        frag = ExperienceFragment("HOW")
        frag.data = {"steps": structured}
        return frag

# 构建 CHECK 规则
class CheckBuilder:
    @staticmethod
    def from_rules(rules: List[str]) -> ExperienceFragment:
        prompt = f"""
你是一个流程规则生成器。请根据以下任务信息，生成流程中可能需要的校验规则，覆盖数据格式、安全性、预算控制等方面。
任务信息：{rules}
返回 JSON 数组格式
"""
        try:
            import json
            response = call_openai(prompt)
            structured = json.loads(response)
        except:
            structured = rules

        frag = ExperienceFragment("CHECK")
        frag.data = {"rules": structured}
        return frag


# -----------------------------
# Stage 3: 经验体演化与质量检验
# -----------------------------
class ExperienceEvaluator:
    @staticmethod
    def update_trust(ep: ExperiencePack, success: bool, feedback_score: float):
        delta = 0.1 if success else -0.1
        ep.trust_score = max(0, min(1.0, ep.trust_score + delta * feedback_score))
        return ep.trust_score

    @staticmethod
    def evolve(ep: ExperiencePack, feedback_summary: str):
        # 调用LLM根据反馈对经验体进行更新
        prompt = f"以下是用户反馈或错误日志，请根据它改进原有经验体结构：\n{feedback_summary}\n原经验体摘要：{ep.summarize()}"
        updated_info = call_openai(prompt)
        frag = ExperienceFragment("WHY")
        frag.data = {"evolution": updated_info}
        ep.version += 1
        ep.add_fragment(frag)
        return ep


# -----------------------------
# Stage 4: 经验激活与工作流生成
# -----------------------------
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


# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    # Stage 1: 收集用户输入
    input_module = MultiModalInput()
    input_module.ingest_interview("我要做一个618广告活动")
    input_module.ingest_interview("因为活动多且时间紧，想自动化")
    input_module.ingest_interview("预算不能超过500元")
    input_module.record_behavior({"page": "campaign", "action": "click", "element": "#create-btn", "timestamp": "T1"})

    # Stage 2: 构建经验体
    ep = ExperiencePack("营销活动流程自动化")
    ep.add_fragment(WhyBuilder.from_interview(input_module.interview_logs))
    ep.add_fragment(HowBuilder.from_behavior(input_module.behavior_logs))
    ep.add_fragment(CheckBuilder.from_rules(["budget <= 500"]))

    # Stage 3: 评估调用后质量
    ExperienceEvaluator.update_trust(ep, success=True, feedback_score=1.0)
    ExperienceEvaluator.evolve(ep, feedback_summary="用户反馈某些流程存在重复操作，建议简化")

    # Stage 4: 构建工作流
    matched = ExperienceRetriever([ep]).match_by_task("营销")
    workflow = WorkflowBuilder.build_from_fragments(matched[0].fragments)
    print("[Workflow Execution Plan]", workflow)
