# extraction_agent.py
"""
Agent: ExtractionAgent
作用：从结构化的观察数据中提取客观知识点（页面结构、操作要素等），并更新经验体的知识图谱。
"""

from typing import List, Dict
from experienceagent.knowledge_graph import KnowledgePoint, ExperienceGraph

class ExtractionAgent:
    def __init__(self, graph: ExperienceGraph):
        self.graph = graph

    def extract_from_observation(self, observation_data: List[Dict]):
        """
        输入：观察数据，格式如 [{"page": "login.html", "element": "input#email", "action": "click"}, ...]
        输出：提取结构化客观知识点并加入图谱
        """
        for obs in observation_data:
            page = obs.get("page")
            element = obs.get("element")
            action = obs.get("action")

            if page and element:
                key = f"OBS_{page}_{element}"
                description = f"在页面 {page} 上存在元素 {element}，支持操作 {action}"
                kp = KnowledgePoint(name=key, content=description, type_="objective", url=page)
                self.graph.add_kp(kp)
        return self.graph

    def extract_pages(self, observation_data: List[Dict]):
        """
        可选功能：从观察数据中仅提取页面级别的存在信息
        """
        seen_pages = set()
        for obs in observation_data:
            page = obs.get("page")
            if page and page not in seen_pages:
                seen_pages.add(page)
                kp = KnowledgePoint(name=f"PAGE_{page}", content=f"页面 {page} 存在", type_="objective", url=page)
                self.graph.add_kp(kp)
        return self.graph
