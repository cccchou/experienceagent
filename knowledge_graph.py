# knowledge_graph.py
from typing import Dict, List, Tuple, Any

class KnowledgePoint:
    def __init__(
        self,
        kid: str,
        content: str,
        ktype: str,  # 'objective', 'subjective', 'domain'
        url: str = "",
        domain: str = ""
    ):
        self.kid = kid
        self.content = content
        self.ktype = ktype
        self.url = url
        self.domain = domain
        self.relations: List[Tuple[str, str]] = []  # (target_kid, relation_type)

    def add_relation(self, target_kid: str, relation_type: str):
        self.relations.append((target_kid, relation_type))


class ExperienceGraph:
    def __init__(self, eid: str):
        self.eid = eid
        self.knowledge_points: Dict[str, KnowledgePoint] = {}

    def add_kp(self, kp: KnowledgePoint):
        self.knowledge_points[kp.kid] = kp

    def add_relation(self, from_kid: str, to_kid: str, rel_type: str):
        if from_kid in self.knowledge_points:
            self.knowledge_points[from_kid].add_relation(to_kid, rel_type)

    def export_as_json(self) -> Dict[str, Any]:
        return {
            "eid": self.eid,
            "knowledge_points": [
                {
                    "kid": kp.kid,
                    "content": kp.content,
                    "ktype": kp.ktype,
                    "url": kp.url,
                    "domain": kp.domain,
                    "relations": kp.relations
                }
                for kp in self.knowledge_points.values()
            ]
        }

    def visualize_console(self):
        print(f"\n[Knowledge Graph: {self.eid}]")
        for kid, kp in self.knowledge_points.items():
            print(f"🔸 [{kp.ktype.upper()}] {kp.kid}: {kp.content}")
            for tgt, rel in kp.relations:
                print(f"    └─({rel})→ {tgt}")


# Example usage
if __name__ == "__main__":
    graph = ExperienceGraph("Exp_T1")

    # 客观知识点
    k1 = KnowledgePoint("K001", "按钮ID为#submit-btn", "objective", url="/submit", domain="页面结构")
    k2 = KnowledgePoint("K002", "页面结构频繁变化", "objective", url="/layout", domain="DOM观察")

    # 主观知识点
    s1 = KnowledgePoint("S001", "验证覆盖率需≥90%", "subjective", domain="质量保障")

    # 领域知识点
    d1 = KnowledgePoint("D001", "所有配置型任务需支持结构化回放验证", "domain", domain="验证通用策略")

    # 添加节点
    for kp in [k1, k2, s1, d1]:
        graph.add_kp(kp)

    # 构建关系
    graph.add_relation("S001", "K002", "依赖")
    graph.add_relation("S001", "D001", "继承")

    # 可视化
    graph.visualize_console()

    # 导出结构化结果
    import json
    print("\n[Export JSON]")
    print(json.dumps(graph.export_as_json(), indent=2, ensure_ascii=False))
