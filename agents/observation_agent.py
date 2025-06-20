# observation_agent.py
"""
ObservationAgent: 用于监听用户在系统内的多模行为（点击、输入、上传等），
并进行实时归类与结构化。
"""
from typing import List, Dict, Any
import json
from openai import OpenAI

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
class ObservationAgent:
    def __init__(self):
        self.raw_logs: List[Dict[str, Any]] = []

    def record(self, behavior_log: Dict[str, Any]):
        self.raw_logs.append(behavior_log)

    def summarize(self) -> List[Dict[str, Any]]:
        if not self.raw_logs:
            return []

        prompt = f"""
你是用户行为观察员，请将以下用户系统行为记录归类整理为结构化步骤：
每一步包括：页面（page）、操作动作（action）、操作对象（element）、初步意图（intent）。
行为记录如下：
{json.dumps(self.raw_logs, indent=2)}
输出格式为 JSON 数组。
"""
        try:
            response = call_openai(prompt)
            structured = json.loads(response)
        except:
            structured = self.raw_logs  # fallback: 原始行为日志

        return structured

# Example usage:
if __name__ == "__main__":
    agent = ObservationAgent()
    agent.record({"page": "dashboard", "action": "click", "element": "#start-btn", "timestamp": "T1"})
    agent.record({"page": "form", "action": "input", "element": "#username", "timestamp": "T2"})
    output = agent.summarize()
    print("[Observation Output]", json.dumps(output, indent=2, ensure_ascii=False))
