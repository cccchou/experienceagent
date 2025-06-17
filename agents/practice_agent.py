# fusion_agent.py
"""
Agent: FusionAgent
作用：融合多个 Agent 提取的主观/客观知识片段，建立统一的知识图谱结构，支持冲突检测、补全与上下位关系链接。
"""

from typing import List
from experienceagent.knowledge_graph import KnowledgePoint, ExperienceGraph

class FusionAgent:
    def __init__(self, graph: ExperienceGraph):
        self.graph = graph

    def fuse(self, fragments: List[List[KnowledgePoint]]) -> None:
        """
        将多个 Agent 的知识点融合到主图谱中。
        """
        for frag in fragments:
            for kp in frag:
                self._try_add_kp(kp)

    def _try_add_kp(self, kp: KnowledgePoint):
        """
        判断是否冲突，若冲突则采用更新策略（保留更可信或新数据）；否则加入图谱
        """
        existing = self.graph.get_kp(kp.name)
        if existing:
            # 简单策略：更新为内容更长的或类型为客观的（可定制）
            if len(kp.content) > len(existing.content) or kp.type == "objective":
                self.graph.update_kp(kp.name, kp)
        else:
            self.graph.add_kp(kp)

    def link_hierarchy(self, parent_name: str, child_names: List[str]):
        """
        构建上下位层级关系
        """
        for child in child_names:
            if self.graph.has_kp(parent_name) and self.graph.has_kp(child):
                self.graph.link(parent_name, child, relation="包含")

    def resolve_conflict(self):
        """
        冲突处理策略（示意）：若同名知识点存在多个内容来源，记录为候选，供后续评分选择
        """
        conflict_map = self.graph.detect_conflicts()
        for name, candidates in conflict_map.items():
            best_kp = max(candidates, key=lambda kp: len(kp.content))
            self.graph.update_kp(name, best_kp)
