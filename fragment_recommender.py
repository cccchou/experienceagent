# fragment_recommender.py

from typing import List, Dict, Any
from openai import OpenAI
import json

client = OpenAI()

def call_openai(prompt: str) -> str:
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "你是一个经验推荐专家，请根据已有的经验片段推荐最相关的协同片段。"},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content

class FragmentRecommender:
    def __init__(self, repository: List[Any]):
        self.repo = repository  # List[ExperiencePack]

    def recommend_fragments(self, query_task: str, query_goal: str, top_k=3) -> List[Dict]:
        """
        从已有经验体中推荐与 query_goal 类似的经验片段（WHY 或 HOW 或 CHECK）。
        """
        all_candidates = []
        for pack in self.repo:
            for frag in pack.fragments:
                frag_info = {
                    "task_name": pack.task_name,
                    "version": pack.version,
                    "frag_type": frag.frag_type,
                    "frag_content": frag.data,
                }
                all_candidates.append(frag_info)

        prompt = f"""
请阅读如下任务目标，并从候选经验片段中推荐与其最相近的 {top_k} 个：
目标任务名称：{query_task}
目标任务目标：{query_goal}

候选片段列表如下（格式为 JSON 数组）：
{json.dumps(all_candidates, ensure_ascii=False)}

请返回最相关的 {top_k} 个片段，输出格式如下：
[
  {{
    "task_name": "...",
    "frag_type": "...",
    "reason": "...推荐理由...",
    "frag_content": {{...}}
  }},
  ...
]
"""
        try:
            response = call_openai(prompt)
            recommended = json.loads(response)
        except Exception as e:
            recommended = [{"task_name": "N/A", "frag_type": "N/A", "reason": f"推荐失败：{str(e)}", "frag_content": {}}]

        return recommended


# Example usage
if __name__ == "__main__":
    from goalfy_learning_framework import ExperiencePack, ExperienceFragment

    ep1 = ExperiencePack("页面变化验证")
    f1 = ExperienceFragment("WHY")
    f1.data = {
        "goal": "自动验证营销页面结构变化",
        "background": "页面变化频繁，测试覆盖不足",
        "constraints": ["不能改代码", "支持多端"],
        "expected_outcome": "自动生成验证用例"
    }
    ep1.add_fragment(f1)

    ep2 = ExperiencePack("客服问题应答构建")
    f2 = ExperienceFragment("WHY")
    f2.data = {
        "goal": "构建通用知识应答系统",
        "background": "客服重复回答问题效率低",
        "constraints": ["需结合已有FAQ", "支持嵌入接口调用"],
        "expected_outcome": "自动知识匹配并回复"
    }
    ep2.add_fragment(f2)

    # 构建推荐器
    recommender = FragmentRecommender([ep1, ep2])
    result = recommender.recommend_fragments("广告活动辅助", "需要验证页面元素变化并自动生成验证流程")
    
    print("[推荐结果]")
    for rec in result:
        print(json.dumps(rec, indent=2, ensure_ascii=False))
