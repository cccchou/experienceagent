# fragment_scorer.py

from typing import Dict, Any
import json
from openai import OpenAI

client = OpenAI()

def call_openai(prompt: str) -> str:
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "你是一个经验评分专家，请严格按照评分标准输出 JSON"},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content

class FragmentScorer:
    @staticmethod
    def score_fragment(frag: Any, task_name: str) -> Dict[str, Any]:
        """
        评估一个经验片段的质量，包括结构完整度、清晰性、迁移性、目标对齐等维度。
        """
        try:
            # 构建提示词
            prompt = f"""
你是一个经验评分专家，请对以下任务的一个经验片段进行评分（0-1之间，保留两位小数），并说明理由。

任务名称：{task_name}
片段类型：{frag.frag_type}
片段内容：{json.dumps(frag.data, ensure_ascii=False)}

请从以下方面进行打分（每项 0-1）：
1. 结构完整度（structure_score）：是否覆盖必要元素？
2. 信息清晰度（clarity_score）：语言是否简洁清晰？
3. 可迁移性（transferability_score）：是否可泛化到类似任务？
4. 目标对齐度（goal_alignment_score）：是否与任务目标紧密相关？

请输出如下 JSON 格式：
{{
  "structure_score": 0.xx,
  "clarity_score": 0.xx,
  "transferability_score": 0.xx,
  "goal_alignment_score": 0.xx,
  "final_score": 0.xx,
  "reasoning": "你的综合评分理由"
}}
"""
            response = call_openai(prompt)
            result = json.loads(response)
        except Exception as e:
            result = {
                "structure_score": 0.5,
                "clarity_score": 0.5,
                "transferability_score": 0.5,
                "goal_alignment_score": 0.5,
                "final_score": 0.5,
                "reasoning": f"默认评分，解析失败：{str(e)}"
            }

        return result


# Example Usage
if __name__ == "__main__":
    from goalfy_learning_framework import ExperienceFragment

    sample_frag = ExperienceFragment("WHY")
    sample_frag.data = {
        "goal": "提升618广告点击率",
        "background": "活动期间竞争激烈，需优化广告文案",
        "constraints": ["预算500元", "时间48小时"],
        "expected_outcome": "点击率提升30%"
    }

    task_name = "618广告优化"
    result = FragmentScorer.score_fragment(sample_frag, task_name)
    print("[评分结果]")
    for k, v in result.items():
        print(f"{k}: {v}")
