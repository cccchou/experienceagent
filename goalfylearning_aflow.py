#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GoalFy Learning Experience Agent with AFlow Dynamic Workflow
Transforms fixed Q&A pattern into adaptive, intelligent conversation flow
"""

import json
import os
import argparse
from openai import OpenAI
from experienceagent.fragment_recommender import ExperienceRetriever, FragmentRecommender
from experienceagent.fragment_scorer import ExperienceEvaluator
from experienceagent.controller_agent import ControllerAgent
from experienceagent.aflow_workflow import ConversationWorkflow
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GoalFyLearning")


class AdaptiveGoalFyAgent:
    """
    Enhanced GoalFy Learning Agent with AFlow dynamic workflow capabilities
    """
    
    def __init__(self, db_path: str = "rich_expert_validation.json", use_dynamic_workflow: bool = True):
        self.db_path = db_path
        self.use_dynamic_workflow = use_dynamic_workflow
        
        # Initialize core components
        self.evaluator = ExperienceEvaluator()
        self.retriever = ExperienceRetriever(db_path, self.evaluator)
        self.recommender = FragmentRecommender(self.retriever, self.evaluator)
        self.controller_agent = ControllerAgent(db_path=db_path)
        
        # Initialize AFlow workflow
        if self.use_dynamic_workflow:
            self.conversation_workflow = ConversationWorkflow()
        
        # Traditional fixed questions (fallback)
        self.fixed_questions = [
            '请问你的目标是什么?',
            '你为什么需要这个功能?',
            '有哪些限制条件我们要考虑?',
            '你希望最终达到什么样的效果?'
        ]
    
    def run_conversation(self) -> Dict[str, Any]:
        """Run the conversation using appropriate method"""
        print("\n====== GoalFy Learning Experience Agent ======")
        print("欢迎使用GoalFy学习体验智能体!")
        
        if self.use_dynamic_workflow:
            print("正在启动AFlow智能对话系统...")
            return self._run_dynamic_workflow()
        else:
            print("使用传统问答模式...")
            return self._run_fixed_questions()
    
    def _run_dynamic_workflow(self) -> Dict[str, Any]:
        """Run the AFlow dynamic workflow conversation"""
        try:
            # Execute the conversation workflow
            workflow_result = self.conversation_workflow.run()
            
            print(f"\n🎯 生成的任务描述: {workflow_result['task_description']}\n")
            
            return {
                "method": "dynamic_workflow",
                "task_description": workflow_result["task_description"],
                "collected_answers": workflow_result["collected_answers"],
                "conversation_history": workflow_result["conversation_history"],
                "workflow_metadata": {
                    "domain_info": workflow_result["context_state"]["domain_info"],
                    "task_type": workflow_result["context_state"]["task_type"],
                    "question_count": workflow_result["context_state"]["question_count"]
                }
            }
        
        except Exception as e:
            logger.error(f"Dynamic workflow failed: {str(e)}, falling back to fixed questions")
            return self._run_fixed_questions()
    
    def _run_fixed_questions(self) -> Dict[str, Any]:
        """Run the traditional fixed questions approach"""
        answers = []
        
        print("告诉我您的需求，我会自动搜索并生成相关经验！")
        for i, question in enumerate(self.fixed_questions, 1):
            print(f"\n[问题 {i}]")
            answer = input(f"{question}\n> ")
            answers.append(answer)
        
        # Generate task description using traditional method
        task_description = self._summarize_task_traditional(answers)
        
        return {
            "method": "fixed_questions",
            "task_description": task_description,
            "collected_answers": answers,
            "conversation_history": [
                {"question": q, "answer": a, "turn": i+1} 
                for i, (q, a) in enumerate(zip(self.fixed_questions, answers))
            ]
        }
    
    def _summarize_task_traditional(self, answers: List[str]) -> str:
        """Traditional task summarization method"""
        try:
            client = OpenAI()
            prompt = (
                "请将以下用户对话回答整合为一句详细的任务描述，用于启动智能体会话。"
                "仅输出一句话的纯文本，末尾不加句号，不要添加任何引号或其他符号，也不要输出多余的注释或解释：\n"
                + "\n".join(f"- {ans}" for ans in answers)
            )
            
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个助理，将用户的回答总结成简练但信息完整的任务描述。"},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Failed to generate task summary: {str(e)}")
            return " ".join(answers[:3])  # Fallback
    
    def run_complete_session(self) -> Dict[str, Any]:
        """Run complete session including conversation and agent processing"""
        try:
            # Run conversation
            conversation_result = self.run_conversation()
            
            task_description = conversation_result["task_description"]
            collected_answers = conversation_result["collected_answers"]
            
            # Initialize and run controller agent
            session = self.controller_agent.new_session(task_description)
            print(f"📋 会话开始: {session['message']}\n")
            
            # Run agent processing
            agent_result = self.controller_agent.run(collected_answers)
            print("🔍 正在处理和分析您的需求...\n")
            
            # Display results
            self._display_recommendations(agent_result)
            
            # Save session
            save_result = self.controller_agent.save_session(data=agent_result)
            print(f"💾 保存会话结果: {save_result['message']}")
            
            self.controller_agent.save_to_file()
            print("📁 已保存到 shuchu.json")
            
            # Enhance knowledge graph
            knowledge_result = self.controller_agent.enhance_knowledge(force_rebuild=True)
            print(f"🧠 知识图谱增强结果: {knowledge_result['message']}")
            
            return {
                "conversation_result": conversation_result,
                "agent_result": agent_result,
                "session_info": session,
                "save_result": save_result,
                "knowledge_result": knowledge_result
            }
            
        except Exception as e:
            logger.error(f"Error in complete session: {str(e)}")
            raise
    
    def _display_recommendations(self, agent_result: Dict[str, Any]):
        """Display agent recommendations in a formatted way"""
        try:
            print("🎯 === 推荐结果 ===")
            
            # Display WHY recommendations
            if 'WHY' in agent_result and agent_result['WHY']:
                print("\n📝 目标与背景相关:")
                for i, item in enumerate(agent_result['WHY'], 1):
                    fragment_data = item.get('fragment', {}).get('data', {})
                    goal = fragment_data.get('goal', '未知目标')
                    similarity = item.get('similarity', 0)
                    source = '🤖 AI生成' if item.get('source') == 'ai_generated' else '📚 经验库'
                    reason = item.get('reason', '无')
                    
                    print(f"  {i}. 任务: {goal}")
                    print(f"     相似度: {similarity:.2f}")
                    print(f"     来源: {source}")
                    print(f"     原因: {reason}")
                    print()
            
            # Display HOW recommendations if available
            if 'HOW' in agent_result and agent_result['HOW']:
                print("🛠️ 实施方法相关:")
                for i, item in enumerate(agent_result['HOW'], 1):
                    fragment_data = item.get('fragment', {}).get('data', {})
                    steps = fragment_data.get('steps', [])
                    source = '🤖 AI生成' if item.get('source') == 'ai_generated' else '📚 经验库'
                    
                    print(f"  {i}. 实施步骤 ({source}):")
                    for j, step in enumerate(steps[:3], 1):  # Show first 3 steps
                        if isinstance(step, dict):
                            action = step.get('action', '')
                            element = step.get('element', '')
                            step_text = f"{action} {element}".strip()
                        else:
                            step_text = str(step)
                        print(f"     {j}. {step_text}")
                    if len(steps) > 3:
                        print(f"     ... 以及其他 {len(steps) - 3} 个步骤")
                    print()
            
            # Display CHECK recommendations if available
            if 'CHECK' in agent_result and agent_result['CHECK']:
                print("✅ 验证检查相关:")
                for i, item in enumerate(agent_result['CHECK'], 1):
                    fragment_data = item.get('fragment', {}).get('data', {})
                    rules = fragment_data.get('rules', [])
                    source = '🤖 AI生成' if item.get('source') == 'ai_generated' else '📚 经验库'
                    
                    print(f"  {i}. 检查规则 ({source}):")
                    for j, rule in enumerate(rules[:3], 1):  # Show first 3 rules
                        print(f"     {j}. {rule}")
                    if len(rules) > 3:
                        print(f"     ... 以及其他 {len(rules) - 3} 个规则")
                    print()
                    
        except Exception as e:
            logger.error(f"Error displaying recommendations: {str(e)}")
            print("❌ 显示推荐结果时出现错误")


def run(db_path: str, use_dynamic_workflow: bool = True):
    """Main execution function"""
    try:
        # Initialize agent
        agent = AdaptiveGoalFyAgent(db_path=db_path, use_dynamic_workflow=use_dynamic_workflow)
        
        # Run complete session
        result = agent.run_complete_session()
        
        # Show completion message
        print("\n🎉 === 任务完成 ===")
        method = result["conversation_result"]["method"]
        if method == "dynamic_workflow":
            metadata = result["conversation_result"]["workflow_metadata"]
            print(f"📊 对话统计: {metadata['question_count']} 个问题")
            print(f"🏷️ 识别领域: {metadata['domain_info'] or '通用'}")
            print(f"📋 任务类型: {metadata['task_type'] or '未知'}")
        else:
            print("📊 使用传统固定问答模式")
        
        input("\n按回车键退出程序...")
        return result
        
    except KeyboardInterrupt:
        print("\n\n👋 用户中断，程序退出")
        return None
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        print(f"\n❌ 程序执行出错: {str(e)}")
        return None


def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description="GoalFy Learning Experience Agent with AFlow")
    parser.add_argument("--db_path", default="rich_expert_validation.json", help="经验库路径")
    parser.add_argument("--disable-dynamic", action="store_true", help="禁用动态工作流，使用传统问答")
    parser.add_argument("--mode", choices=["dynamic", "fixed"], default="dynamic", help="对话模式选择")
    
    args = parser.parse_args()
    
    # Determine workflow mode
    use_dynamic_workflow = not args.disable_dynamic and args.mode == "dynamic"
    
    if use_dynamic_workflow:
        print("🚀 启动AFlow动态工作流模式")
    else:
        print("📋 启动传统固定问答模式")
    
    run(db_path=args.db_path, use_dynamic_workflow=use_dynamic_workflow)


if __name__ == "__main__":
    main()