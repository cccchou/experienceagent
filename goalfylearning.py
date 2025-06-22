#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GoalFy Learning Experience Agent - 简洁对话式体验
获取基本任务信息后自动搜索和生成相关经验
修复版本：解决shuchu.json输出格式问题，确保完整信息导出
"""

import os
import sys
import json
import time
import argparse
import re
from typing import Dict, List, Optional, Tuple
from experienceagent.controller_agent import ControllerAgent, ExperienceFragment

class DialogueState:
    """简化的对话状态管理类"""
    
    def __init__(self):
        self.why_data = {  # WHY片段数据
            "goal": "",
            "background": "",
            "constraints": [],
            "expected_outcome": ""
        }
        self.current_question = "task"  # 当前问题类型
        self.task_name = ""
        self.dialogue_logs = []  # 对话日志
        self.has_basic_info = False  # 是否已收集基本信息
        self.question_asked = {}  # 跟踪已询问的问题
    
    def add_dialogue(self, role: str, message: str) -> None:
        """添加对话记录"""
        self.dialogue_logs.append(f"{role}：{message}")
    
    def get_next_question(self) -> Optional[str]:
        """获取下一个问题"""
        # 检查是否已收集足够信息
        if self.has_enough_info():
            self.has_basic_info = True
            return None
            
        if not self.task_name:
            self.current_question = "task"
            self.question_asked["task"] = True
            return "请简要描述您想要解决的问题或实现的功能。"
            
        if not self.why_data["goal"] and not self.question_asked.get("goal"):
            self.current_question = "goal"
            self.question_asked["goal"] = True
            return "您的目标是什么？您希望实现什么功能？"
            
        if not self.why_data["background"] and not self.question_asked.get("background"):
            self.current_question = "background"
            self.question_asked["background"] = True
            return "这个功能解决了什么问题？实现背景是什么？"
            
        if not self.why_data["constraints"] and not self.question_asked.get("constraints"):
            self.current_question = "constraints"
            self.question_asked["constraints"] = True
            return "在实现过程中，有什么约束条件或限制需要考虑？"
            
        if not self.why_data["expected_outcome"] and not self.question_asked.get("expected_outcome"):
            self.current_question = "expected_outcome"
            self.question_asked["expected_outcome"] = True
            return "您期望达到什么效果？如何评价成功与否？"
        
        # 如果所有问题都已询问但有些没有回答，也视为完成
        if all(self.question_asked.get(q) for q in ["task", "goal", "background", "constraints", "expected_outcome"]):
            self.has_basic_info = True
            return None
            
        # 所有基本信息已收集
        self.has_basic_info = True
        return None
    
    def has_enough_info(self) -> bool:
        """检查是否已收集足够信息"""
        # 至少需要任务名称和目标
        if not self.task_name or not self.why_data["goal"]:
            return False
            
        # 检查是否已收集至少3个WHY字段
        filled_count = sum(1 for k, v in self.why_data.items() if v)
        if filled_count >= 3:
            return True
            
        return False
    
    def process_input(self, user_input: str) -> str:
        """处理用户输入"""
        # 记录用户输入
        self.add_dialogue("用户", user_input)
        
        # 提取综合信息
        extracted_info = self._extract_info(user_input)
        
        # 根据当前问题处理输入
        if self.current_question == "task":
            self.task_name = extracted_info.get("task", user_input)
            response = f"了解，您需要实现「{self.task_name}」。"
        elif self.current_question == "goal":
            self.why_data["goal"] = extracted_info.get("goal", user_input)
            response = "目标已记录。"
        elif self.current_question == "background":
            self.why_data["background"] = extracted_info.get("background", user_input)
            response = "背景已记录。"
        elif self.current_question == "constraints":
            # 从输入提取多个约束
            constraints = extracted_info.get("constraints", [])
            if not constraints:
                constraints = self._extract_list_items(user_input)
            self.why_data["constraints"] = constraints
            response = f"已记录{len(constraints)}个约束条件。"
        elif self.current_question == "expected_outcome":
            self.why_data["expected_outcome"] = extracted_info.get("expected_outcome", user_input)
            response = "预期效果已记录。"
        else:
            response = "感谢您的输入。"
            
        # 记录系统响应
        self.add_dialogue("系统", response)
        
        # 检查是否自动提取了其他信息
        self._auto_fill_from_extracted(extracted_info)
        
        # 检查是否有足够信息
        if self.has_enough_info():
            self.has_basic_info = True
            
        return response
    
    def _extract_info(self, text: str) -> Dict:
        """从输入中提取各种信息"""
        info = {
            "task": "",
            "goal": "",
            "background": "",
            "constraints": [],
            "expected_outcome": ""
        }
        
        # 尝试提取任务
        if "实现" in text or "开发" in text or "创建" in text:
            task_match = re.search(r'(实现|开发|创建)(一个|一套)?([^，。]+)(系统|框架|工具|功能|平台)?', text)
            if task_match:
                info["task"] = task_match.group(0)
        
        # 尝试提取目标
        if "目标" in text or "希望" in text or "要做到" in text:
            goal_match = re.search(r'目标(是|：|:)?([^。]+)(?=。|$)', text)
            if goal_match:
                info["goal"] = goal_match.group(2).strip()
            elif not info["goal"]:  # 如果没有通过明确的"目标"提取到，使用整个输入作为目标
                info["goal"] = text
        
        # 尝试提取背景
        if "背景" in text or "因为" in text or "由于" in text:
            bg_match = re.search(r'(背景|原因)(是|：|:)?([^。]+)(?=。|$)', text)
            if bg_match:
                info["background"] = bg_match.group(3).strip()
            else:
                bg_match = re.search(r'(因为|由于)([^。]+)(?=。|$)', text)
                if bg_match:
                    info["background"] = bg_match.group(0)
        
        # 尝试提取约束
        if "约束" in text or "限制" in text or "条件" in text:
            const_match = re.search(r'(约束|限制|条件)(是|包括|：|:)?([^。]+)(?=。|$)', text)
            if const_match:
                constraints_text = const_match.group(3).strip()
                info["constraints"] = self._extract_list_items(constraints_text)
        
        # 尝试提取预期效果
        if "预期" in text or "效果" in text or "结果" in text or "评价" in text:
            outcome_match = re.search(r'(预期|效果|结果|评价)(是|：|:)?([^。]+)(?=。|$)', text)
            if outcome_match:
                info["expected_outcome"] = outcome_match.group(3).strip()
        
        return info
    
    def _extract_list_items(self, text: str) -> List[str]:
        """从文本中提取列表项"""
        items = []
        
        # 按分隔符拆分
        parts = re.split(r'[,，;；、]', text)
        items = [part.strip() for part in parts if part.strip()]
        
        # 如果只有一项，返回原文本
        if not items or (len(items) == 1 and not items[0]):
            items = [text.strip()]
            
        return items
    
    def _auto_fill_from_extracted(self, info: Dict) -> None:
        """自动填充从用户输入中提取的额外信息"""
        # 只自动填充尚未填写的字段
        if info.get("goal") and not self.why_data["goal"]:
            self.why_data["goal"] = info["goal"]
            
        if info.get("background") and not self.why_data["background"]:
            self.why_data["background"] = info["background"]
            
        if info.get("constraints") and not self.why_data["constraints"]:
            self.why_data["constraints"] = info["constraints"]
            
        if info.get("expected_outcome") and not self.why_data["expected_outcome"]:
            self.why_data["expected_outcome"] = info["expected_outcome"]


class GoalFyLearningAgent:
    """GoalFy学习智能体，提供简洁对话式经验学习和推荐"""
    
    def __init__(self, db_path: str = "rich_expert_validation.json"):
        """
        初始化GoalFy学习智能体
        
        Args:
            db_path: 经验库路径
        """
        print("初始化GoalFy学习智能体...")
        self.controller = ControllerAgent(db_path)
        self.dialogue_state = DialogueState()
        self.started_search = False  # 是否已开始搜索
        self.collected_fragments = {  # 收集的所有片段
            "WHY": None,
            "HOW": None,
            "CHECK": None
        }
        
        # 确认经验库存在
        if not os.path.exists(db_path):
            print(f"警告: 经验库文件不存在 ({db_path})，将创建新的经验库")
            # 创建空的经验库文件
            with open(db_path, "w", encoding="utf-8") as f:
                json.dump({}, f)
        
        print(f"已加载经验库: {db_path}")
        
        # 生成shuchu.json的默认占位符
        if not os.path.exists("shuchu.json"):
            placeholder = {
                "task": "初始化任务",
                "version": 1,
                "trust_score": 0.5,
                "fragments": [],
                "workflow_plan": {"steps": []}
            }
            with open("shuchu.json", "w", encoding="utf-8") as f:
                json.dump(placeholder, f, ensure_ascii=False, indent=2)
                
        # 开始新会话
        self.start_session()
    
    def start_session(self) -> None:
        """启动新会话"""
        # 系统提示
        print("\n====== GoalFy Learning Experience Agent ======")
        print("欢迎使用GoalFy学习体验智能体!")
        print("告诉我您的需求，我会自动搜索并生成相关经验！")
        print("输入'退出'结束对话，输入'跳过'直接开始搜索\n")
        
        # 获取初始任务
        task_name = f"学习任务 {time.strftime('%Y-%m-%d %H:%M')}"
        result = self.controller.new_session(task_name)
        
        # 获取第一个问题
        first_question = self.dialogue_state.get_next_question()
        
        # 通知用户
        print(f"系统: {first_question}")
    
    def search_and_recommend(self) -> None:
        """自动搜索和推荐相关经验"""
        # 标记已开始搜索
        self.started_search = True
        
        # 准备WHY片段
        why_data = self.dialogue_state.why_data
        
        # 确保WHY数据不为空
        if not why_data["goal"]:
            why_data["goal"] = self.dialogue_state.task_name or "未指定目标"
        
        # 记录WHY片段
        self.collected_fragments["WHY"] = why_data
        
        # 添加到控制器
        self.controller.add_fragment("WHY", why_data)
        
        # 更新任务名称
        if self.dialogue_state.task_name:
            self.controller.current_experience.task_name = self.dialogue_state.task_name
        
        # 构建搜索查询
        query = f"{self.dialogue_state.task_name}: {why_data['goal']}"
        
        print("\n系统: 已收集基本信息，正在为您搜索相关经验...")
        
        # 获取经验推荐
        result = self.controller.recommend_fragments(query)
        
        # 显示是否使用了AI生成
        if result.get("has_ai_generated"):
            print("\n系统: 由于经验库中没有完全匹配的内容，我已使用AI智能生成了补充内容。")
        
        # 处理推荐结果
        if result["has_recommendations"]:
            print("\n系统: 已找到以下相关经验:")
            
            # 自动采纳高相关性的推荐
            for frag_type in ["HOW", "CHECK"]:
                if frag_type in result["recommendations"]:
                    items = result["recommendations"][frag_type]
                    if items:
                        best_item = items[0]
                        source_type = "AI智能生成" if best_item.get("source") == "ai_generated" else "经验库"
                        
                        # 显示推荐内容摘要
                        print(f"\n- 采纳{frag_type}类型经验 ({source_type}):")
                        if frag_type == "HOW" and "steps" in best_item['content']:
                            steps = best_item['content']['steps']
                            print(f"  共{len(steps)}个步骤:")
                            for i, step in enumerate(steps[:3]):  # 显示前3步
                                if isinstance(step, dict):
                                    print(f"  {i+1}. {step.get('action', '')} {step.get('element', '')}")
                                    if i == 2 and len(steps) > 3:
                                        print(f"  ... 等共{len(steps)}步")
                        elif frag_type == "CHECK" and "rules" in best_item['content']:
                            rules = best_item['content']['rules']
                            print(f"  共{len(rules)}条规则:")
                            for i, rule in enumerate(rules[:3]):  # 显示前3条规则
                                print(f"  {i+1}. {rule}")
                                if i == 2 and len(rules) > 3:
                                    print(f"  ... 等共{len(rules)}条")
                        
                        # 记录片段内容
                        self.collected_fragments[frag_type] = best_item['content']
                        
                        # 自动采纳
                        self.controller.add_fragment(frag_type, best_item['content'])
            
            # 确保数据完整，生成缺失的片段
            self._ensure_complete_data()
            
            # 更新工作流计划
            self._update_workflow_plan()
            
            # 保存到shuchu.json
            self._save_complete_output()
            
            # 保存到经验库
            self._save_experience_with_dialogue()
            
            print("\n系统: 已自动为您完善经验内容并保存。需要查看完整经验吗？(是/否)")
            choice = input("用户: ").lower()
            if choice in ["是", "yes", "y", "需要", "查看"]:
                self._show_complete_experience()
            
            print("\n系统: 您还有其他问题吗？或者输入'退出'结束对话。")
        else:
            print("\n系统: 抱歉，未找到完全匹配的经验。我将尝试生成基础内容...")
            # 请求AI直接生成HOW和CHECK内容
            self._generate_missing_content()
    
    def _ensure_complete_data(self) -> None:
        """确保数据完整性"""
        # 检查并生成缺失的片段
        for frag_type in ["HOW", "CHECK"]:
            if not self.collected_fragments.get(frag_type):
                print(f"\n系统: 未找到相关的{frag_type}片段，正在智能生成...")
                self._generate_specific_fragment(frag_type)
    
    def _generate_specific_fragment(self, frag_type: str) -> None:
        """生成特定类型的片段"""
        task = self.dialogue_state.task_name or "用户任务"
        why_data = self.dialogue_state.why_data
        
        # 生成片段
        result = self.controller.recommender._generate_fragment(
            f"{task}: {why_data['goal']}",
            frag_type,
            f"目标: {why_data['goal']}, 背景: {why_data['background']}"
        )
        
        if result:
            item = result[0]
            data = item["fragment"]["data"]
            
            # 记录片段
            self.collected_fragments[frag_type] = data
            
            # 添加到控制器
            self.controller.add_fragment(frag_type, data)
            
            # 显示生成内容
            print(f"\n- 生成{frag_type}内容 (AI智能生成):")
            if frag_type == "HOW" and "steps" in data:
                steps = data.get("steps", [])
                print(f"  共{len(steps)}个步骤:")
                for i, step in enumerate(steps[:3]):
                    if isinstance(step, dict):
                        print(f"  {i+1}. {step.get('action', '')} {step.get('element', '')}")
            elif frag_type == "CHECK" and "rules" in data:
                rules = data.get("rules", [])
                print(f"  共{len(rules)}条规则:")
                for i, rule in enumerate(rules[:3]):
                    print(f"  {i+1}. {rule}")
    
    def _update_workflow_plan(self) -> None:
        """更新工作流计划"""
        # 从HOW片段提取工作流步骤
        workflow_steps = []
        how_data = self.collected_fragments.get("HOW")
        
        if how_data and "steps" in how_data:
            for step in how_data["steps"]:
                if isinstance(step, dict):
                    action = step.get("action", "")
                    element = step.get("element", "")
                    if action and element:
                        workflow_steps.append(f"{action} {element}")
                    elif action:
                        workflow_steps.append(action)
        
        # 更新工作流计划
        if workflow_steps:
            self.controller.update_workflow_plan(workflow_steps)
    
    def _save_complete_output(self) -> None:
        """保存完整的输出到shuchu.json"""
        # 确保任务名称已设置
        if self.dialogue_state.task_name:
            self.controller.current_experience.task_name = self.dialogue_state.task_name
        
        # 设置信任分数
        self.controller.current_experience.trust_score = 0.5
        
        # 保存到文件
        save_result = self.controller.save_to_file("shuchu.json")
        
        if save_result["success"]:
            print("\n系统: 完整经验已保存到shuchu.json")
        else:
            print(f"\n系统: 保存shuchu.json失败: {save_result['message']}")
    
    def _generate_missing_content(self) -> None:
        """直接调用AI生成缺失内容"""
        # 生成HOW片段
        if not self.collected_fragments.get("HOW"):
            self._generate_specific_fragment("HOW")
        
        # 生成CHECK片段
        if not self.collected_fragments.get("CHECK"):
            self._generate_specific_fragment("CHECK")
        
        # 更新工作流计划
        self._update_workflow_plan()
        
        # 保存到shuchu.json
        self._save_complete_output()
        
        # 保存到经验库
        self._save_experience_with_dialogue()
        
        print("\n系统: 已生成并保存完整经验内容。需要查看完整经验吗？(是/否)")
        choice = input("用户: ").lower()
        if choice in ["是", "yes", "y", "需要", "查看"]:
            self._show_complete_experience()
        
        print("\n系统: 您还有其他问题吗？或者输入'退出'结束对话。")
    
    def _save_experience_with_dialogue(self) -> None:
        """保存经验和对话日志"""
        # 更新任务名称
        if self.dialogue_state.task_name:
            self.controller.current_experience.task_name = self.dialogue_state.task_name
        
        # 准备对话日志
        dialogue_data = {
            "dialogue_logs": self.dialogue_state.dialogue_logs
        }
        
        # 保存到经验库
        self.controller.save_current_experience(
            self.dialogue_state.task_name, 
            json.dumps(dialogue_data)
        )
    
    def _show_complete_experience(self) -> None:
        """展示完整经验内容"""
        if not self.controller.current_experience:
            print("\n系统: 当前没有完整的经验内容。")
            return
            
        # 获取完整经验
        exp_data = self.controller.export_experience("dict")["data"]
        
        print("\n====== 完整经验内容 ======")
        print(f"任务: {exp_data['task']}")
        print(f"版本: {exp_data['version']}")
        print(f"信任分数: {exp_data['trust_score']}")
        
        # 显示各部分内容
        for fragment in exp_data.get("fragments", []):
            frag_type = fragment.get("type", "")
            data = fragment.get("data", {})
            
            if frag_type == "WHY":
                print("\n【WHY - 目标与背景】")
                print(f"目标: {data.get('goal', '')}")
                print(f"背景: {data.get('background', '')}")
                if "constraints" in data and data["constraints"]:
                    print("约束条件:")
                    for i, constraint in enumerate(data["constraints"]):
                        print(f"  {i+1}. {constraint}")
                print(f"预期效果: {data.get('expected_outcome', '')}")
                
            elif frag_type == "HOW":
                print("\n【HOW - 实现步骤】")
                if "steps" in data:
                    for i, step in enumerate(data["steps"]):
                        if isinstance(step, dict):
                            print(f"步骤{i+1}: {step.get('action', '')} {step.get('element', '')}")
                            if step.get("page"):
                                print(f"  页面: {step['page']}")
                            if step.get("intent"):
                                print(f"  目的: {step['intent']}")
                            
            elif frag_type == "CHECK":
                print("\n【CHECK - 验证规则】")
                if "rules" in data:
                    for i, rule in enumerate(data["rules"]):
                        print(f"规则{i+1}: {rule}")
        
        # 显示工作流
        if "workflow_plan" in exp_data and exp_data["workflow_plan"].get("steps"):
            steps = exp_data["workflow_plan"]["steps"]
            print("\n【工作流】")
            for i, step in enumerate(steps):
                print(f"{i+1}. {step}")
    
    def process_input(self, user_input: str) -> bool:
        """
        处理用户输入
        
        Args:
            user_input: 用户输入
            
        Returns:
            是否继续对话
        """
        # 检查退出命令
        if any(cmd in user_input.lower() for cmd in ['退出', 'exit', 'quit', '再见', '结束']):
            # 如果有收集的信息但还未搜索，先进行搜索
            if not self.started_search and (self.dialogue_state.has_basic_info or self.dialogue_state.task_name):
                print("\n系统: 在退出前，让我为您搜索相关经验...")
                self.search_and_recommend()
            
            print("\n系统: 感谢使用GoalFy学习智能体，再见!")
            return False
            
        # 检查跳过命令
        if any(cmd in user_input.lower() for cmd in ['跳过', 'skip', '直接搜索', '开始搜索']):
            print("\n系统: 好的，将跳过剩余问题，直接开始搜索...")
            # 确保有任务名称
            if not self.dialogue_state.task_name:
                self.dialogue_state.task_name = user_input
            # 确保有目标
            if not self.dialogue_state.why_data["goal"]:
                self.dialogue_state.why_data["goal"] = user_input
            
            self.dialogue_state.has_basic_info = True
            self.search_and_recommend()
            return True
        
        # 处理用户输入
        response = self.dialogue_state.process_input(user_input)
        print(f"\n系统: {response}")
        
        # 检查是否已收集基本信息
        next_question = self.dialogue_state.get_next_question()
        
        if self.dialogue_state.has_basic_info and not self.started_search:
            # 基本信息已收集完毕，自动开始搜索
            self.search_and_recommend()
            return True
        elif next_question:
            # 继续提问
            print(f"\n系统: {next_question}")
        else:
            # 如果没有下一个问题且尚未开始搜索，立即开始
            if not self.started_search:
                self.dialogue_state.has_basic_info = True
                self.search_and_recommend()
        
        return True
    
    def run(self) -> None:
        """运行交互式学习智能体"""
        try:
            while True:
                user_input = input("\n用户: ")
                if not user_input.strip():
                    continue
                    
                if not self.process_input(user_input):
                    break
        except KeyboardInterrupt:
            print("\n\n系统: 程序已被中断。")
            # 如果有收集的信息但还未搜索，尝试进行搜索
            if not self.started_search and (self.dialogue_state.task_name or self.dialogue_state.why_data["goal"]):
                choice = input("\n系统: 是否在退出前为您搜索相关经验？(是/否): ").lower()
                if choice in ["是", "yes", "y"]:
                    self.search_and_recommend()
        except Exception as e:
            print(f"\n系统错误: {str(e)}")
            import traceback
            traceback.print_exc()
            # 如果可能，保存已收集信息
            try:
                if self.dialogue_state.task_name or self.dialogue_state.why_data["goal"]:
                    # 确保有目标
                    if not self.dialogue_state.why_data["goal"]:
                        self.dialogue_state.why_data["goal"] = self.dialogue_state.task_name or "未指定目标"
                        
                    self.controller.add_fragment("WHY", self.dialogue_state.why_data)
                    self._save_complete_output()
                    print("\n系统: 已保存您提供的基本信息。")
            except:
                pass


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="GoalFy Learning Experience Agent")
    parser.add_argument("--db", default="rich_expert_validation.json", help="经验库路径")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    
    args = parser.parse_args()
    
    try:
        # 初始化并运行智能体
        agent = GoalFyLearningAgent(args.db)
        agent.run()
    except Exception as e:
        print(f"\n程序发生错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    # 运行主函数
    main()