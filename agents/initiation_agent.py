# initiation_agent.py
"""
Agent Name: InitiationAgent
Purpose: Launch the goalfy learning process by confirming the intent, scope, and feasibility of experience construction.
"""

from typing import List, Dict
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
class InitiationAgent:
    def __init__(self):
        pass

    def run(self, dialogs: List[str]) -> Dict:
        prompt = f"""
你是一个经验体构建的发起代理，请根据以下用户对话，判断是否具备构建任务经验体的前置条件。
用户对话如下：
{chr(10).join(dialogs)}

请输出以下内容：
1. 是否为一次性任务？（one_time: true/false）
2. 是否具有泛化价值？（generalizable: true/false）
3. 任务意图是什么？（goal）
4. 涉及的页面或系统（pages / system）
5. 是否建议继续进行构建（can_proceed: true/false）
请以JSON格式返回。
"""

        import json
        try:
            raw = call_llm(prompt)
            result = json.loads(raw)
        except:
            result = {
                "one_time": False,
                "generalizable": True,
                "goal": dialogs[0] if dialogs else "未定义",
                "pages": ["unknown"],
                "system": "default",
                "can_proceed": True
            }
        return result


if __name__ == "__main__":
    agent = InitiationAgent()
    session_meta = agent.run([
        "我想做一个可以反复复用的活动发布经验模板，适用于每次促销运营。",
        "目标是减少每次上线之前的重复配置与流程设定。"
    ])
    print("[Session Meta]", session_meta)
