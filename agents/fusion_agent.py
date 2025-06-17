# fusion_agent.py
"""
Agent: FusionAgent
功能：融合多个智能体提取的知识片段，构建统一经验图谱 ExperienceGraph。
支持冲突消解、上下位结构关系建立、知识点可信度更新。
"""

from typing import List
from experienceagent.knowledge_graph import KnowledgePoint, ExperienceGraph

class FusionAgent:
    def __init__(self, graph: ExperienceGraph):
        self.graph = graph

    def fuse_knowledge_fragments(self, fragments: List[List[KnowledgePoint]]) -> None:
        """
        输入多个来源（agent）生成的知识点列表，对图谱进行融合处理。
        """
        for fragment in fragments:
            for kp in fragment:
                self._merge_or_update(kp)

    def _merge_or_update(self, new_kp: KnowledgePoint):
        """
        如果知识点已存在则更新其内容或合并信任信息，否则新增。
        """
        existing = self.graph.get_kp(new_kp.name)
        if existing:
            # 简单冲突策略：优先保留“客观”类型或内容较丰富者
            if new_kp.type == "objective" or len(new_kp.content) > len(existing.content):
                self.graph.update_kp(new_kp.name, new_kp)
            else:
                # 可以记录为候选待评分
                self.graph.add_candidate(new_kp.name, new_kp)
        else:
            self.graph.add_kp(new_kp)

    def establish_hierarchy(self, hierarchy: dict):
        """
        构建上下位关系，例如：
        {
            "用户操作流程": ["点击按钮", "填写表单"],
            "营销分析路径": ["获取数据", "识别转化漏斗"]
        }
        """
        for parent, children in hierarchy.items():
            for child in children:
                if self.graph.has_kp(parent) and self.graph.has_kp(child):
                    self.graph.link(parent, child, relation="包含")

    def resolve_conflicts(self):
        """
        冲突消解策略，当前版本采用内容丰富优先，可扩展为 LLM 评分比较
        """
        conflicts = self.graph.detect_conflicts()
        for name, candidates in conflicts.items():
            best = max(candidates, key=lambda kp: len(kp.content))
            self.graph.update_kp(name, best)

    def augment_with_context(self, context_kps: List[KnowledgePoint]):
        """
        注入领域知识点或上下文信息（例如业务规则、行业背景等）
        """
        for kp in context_kps:
            if not self.graph.has_kp(kp.name):
                self.graph.add_kp(kp)
