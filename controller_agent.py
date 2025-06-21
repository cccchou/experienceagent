# experienceagent/controller_agent.py
"""
经验智能体控制器
负责协调经验检索、推荐和学习，匹配shuchu.json格式
"""

import os
import json
import logging
import time
from typing import List, Dict, Any, Optional, Union
from experienceagent.fragment_recommender import ExperienceRetriever, FragmentRecommender
from experienceagent.fragment_scorer import ExperienceEvaluator

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ExperienceAgent")


class ExperienceFragment:
    """经验片段类，用于统一片段表示"""
    
    def __init__(self, frag_type: str, data: Dict):
        self.frag_type = frag_type
        self.data = data
    
    def to_dict(self) -> Dict:
        """转换为字典格式，匹配shuchu.json格式"""
        return {
            "type": self.frag_type,
            "data": self.data
        }


class ExperiencePack:
    """经验包类，用于组织和管理经验，匹配shuchu.json格式"""
    
    def __init__(self, task_name: str):
        self.task_name = task_name
        self.fragments = []
        self.version = 1
        self.trust_score = 0.5  # 默认信任分数
        self.workflow_plan = {"steps": []}  # 添加workflow_plan字段
    
    def add_fragment(self, fragment: ExperienceFragment):
        """添加经验片段"""
        self.fragments.append(fragment)
        
        # 如果是HOW类型片段，自动提取步骤更新workflow_plan
        if fragment.frag_type == "HOW" and "steps" in fragment.data:
            workflow_steps = []
            for step in fragment.data["steps"]:
                if isinstance(step, dict):
                    action = step.get("action", "")
                    element = step.get("element", "")
                    workflow_steps.append(f"{action}{' ' + element if element else ''}")
                else:
                    workflow_steps.append(str(step))
            
            self.workflow_plan["steps"] = workflow_steps
    
    def get_fragment(self, frag_type: str) -> Optional[ExperienceFragment]:
        """获取特定类型的片段"""
        for fragment in self.fragments:
            if fragment.frag_type == frag_type:
                return fragment
        return None
    
    def has_fragment_type(self, frag_type: str) -> bool:
        """检查是否有特定类型的片段"""
        return self.get_fragment(frag_type) is not None
    
    def to_dict(self) -> Dict:
        """转换为字典格式，匹配shuchu.json格式"""
        return {
            "task": self.task_name,
            "version": self.version,
            "trust_score": self.trust_score,
            "fragments": [f.to_dict() for f in self.fragments],
            "workflow_plan": self.workflow_plan
        }


class ControllerAgent:
    """
    经验智能体控制器
    协调经验检索、推荐和学习
    """
    
    def __init__(self, db_path: str = "rich_expert_validation.json"):
        """
        初始化智能体控制器
        
        Args:
            db_path: 经验数据库路径
        """
        self.db_path = db_path
        
        # 初始化评估器
        self.evaluator = ExperienceEvaluator()
        
        # 初始化检索器和推荐器
        self.retriever = ExperienceRetriever(db_path, self.evaluator)
        self.recommender = FragmentRecommender(self.retriever, self.evaluator)
        
        # 会话状态
        self.current_task = None
        self.current_experience = None
        self.context_history = []  # 上下文历史
        self.session_start_time = time.time()
    
    def new_session(self, task_name: str = None) -> Dict:
        """
        开始新的会话
        
        Args:
            task_name: 任务名称
        
        Returns:
            会话信息
        """
        # 保存之前会话的对话历史（如果有）
        if self.recommender.dialogue_history:
            self.recommender.save_dialogue_history()
        
        # 重置会话状态
        self.context_history = []
        self.recommender.dialogue_history = []
        self.session_start_time = time.time()
        
        # 设置新任务
        if task_name:
            self.current_task = task_name
            self.current_experience = ExperiencePack(task_name)
            message = f"已开始「{task_name}」新会话"
        else:
            self.current_task = "未命名任务"
            self.current_experience = ExperiencePack("未命名任务")
            message = "已开始新会话"
        
        return {
            "success": True,
            "message": message,
            "session_id": str(int(self.session_start_time)),
            "experience_pack": self.current_experience.to_dict()
        }
    
    def process_user_input(self, user_input: str, task_hint: str = None) -> Dict:
        """
        处理用户输入
        
        Args:
            user_input: 用户输入
            task_hint: 任务提示
            
        Returns:
            处理结果
        """
        # 记录用户输入到对话历史
        self.recommender.dialogue_history.append(f"用户: {user_input}")
        
        # 添加到上下文历史
        self.context_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": time.time()
        })
        
        # 如果没有当前任务但有任务提示，设置当前任务
        if not self.current_task and task_hint:
            self.current_task = task_hint
            self.current_experience = ExperiencePack(task_hint)
        
        # 根据用户输入寻找相关经验
        search_query = user_input
        if task_hint:
            search_query = f"{task_hint}: {user_input}"
            
        similar_experiences = self.retriever.semantic_search(
            search_query, top_k=3
        )
        
        # 准备推荐内容
        recommendations = {}
        if similar_experiences:
            # 处理rich_expert_validation.json格式的经验
            for exp_item in similar_experiences:
                exp = exp_item["experience"]
                
                # 提取WHY片段
                if "why_structured" in exp and not self.current_experience.has_fragment_type("WHY"):
                    if "WHY" not in recommendations:
                        recommendations["WHY"] = []
                    
                    why_data = {
                        "goal": exp["why_structured"].get("goal", ""),
                        "background": exp["why_structured"].get("background", ""),
                        "constraints": exp["why_structured"].get("constraints", []),
                        "expected_outcome": exp["why_structured"].get("expected_outcome", "")
                    }
                    
                    recommendations["WHY"].append({
                        "content": why_data,
                        "similarity": exp_item["similarity"],
                        "task_name": exp.get("task_title", "未命名任务")
                    })
                
                # 提取HOW片段
                if "how_behavior_logs" in exp and not self.current_experience.has_fragment_type("HOW"):
                    if "HOW" not in recommendations:
                        recommendations["HOW"] = []
                    
                    # 转换为shuchu.json的格式
                    steps = []
                    for log in exp["how_behavior_logs"]:
                        if isinstance(log, dict):
                            steps.append({
                                "page": log.get("page", ""),
                                "action": log.get("action", ""),
                                "element": log.get("element", ""),
                                "intent": log.get("intent", "")
                            })
                    
                    how_data = {"steps": steps}
                    
                    recommendations["HOW"].append({
                        "content": how_data,
                        "similarity": exp_item["similarity"],
                        "task_name": exp.get("task_title", "未命名任务")
                    })
                
                # 提取CHECK片段
                if "check_rules" in exp and not self.current_experience.has_fragment_type("CHECK"):
                    if "CHECK" not in recommendations:
                        recommendations["CHECK"] = []
                    
                    check_data = {"rules": exp["check_rules"]}
                    
                    recommendations["CHECK"].append({
                        "content": check_data,
                        "similarity": exp_item["similarity"],
                        "task_name": exp.get("task_title", "未命名任务")
                    })
        
        # 记录系统响应到对话历史
        response = "我找到了一些相关经验，可以帮助你解决问题。"
        self.recommender.dialogue_history.append(f"系统: {response}")
        
        # 转换结果为shuchu.json格式
        result = {
            "success": True,
            "message": response,
            "recommendations": recommendations,
            "similar_experiences": [
                {
                    "task_name": item["experience"].get("task_title", "未命名任务"),
                    "similarity": item["similarity"],
                    "reason": item["reason"]
                }
                for item in similar_experiences
            ],
            "current_experience": self.current_experience.to_dict() if self.current_experience else None
        }
        
        return result
    
    def add_fragment(self, frag_type: str, fragment_data: Dict, user_input: str = None) -> Dict:
        """
        添加经验片段
        
        Args:
            frag_type: 片段类型 (WHY, HOW, CHECK)
            fragment_data: 片段数据
            user_input: 用户输入上下文
            
        Returns:
            添加结果
        """
        # 确保有当前经验
        if not self.current_experience:
            self.current_experience = ExperiencePack("未命名任务")
        
        # 格式化片段数据，确保符合shuchu.json格式
        formatted_data = self._format_fragment_data(frag_type, fragment_data)
        
        # 创建并添加片段
        fragment = ExperienceFragment(frag_type, formatted_data)
        self.current_experience.add_fragment(fragment)
        
        # 记录上下文
        context = f"添加了{frag_type}类型片段"
        if user_input:
            context = f"{user_input} -> {context}"
            # 添加到对话历史
            self.recommender.dialogue_history.append(f"用户: {user_input}")
        
        self.recommender.dialogue_history.append(f"系统: 已添加{frag_type}类型片段到当前经验")
        
        # 添加到上下文历史
        self.context_history.append({
            "role": "system",
            "content": f"添加了{frag_type}类型片段",
            "timestamp": time.time(),
            "data": {
                "type": frag_type,
                "fragment": formatted_data
            }
        })
        
        return {
            "success": True,
            "message": f"已添加{frag_type}类型片段到当前经验",
            "fragment_type": frag_type,
            "fragment_count": len(self.current_experience.fragments),
            "experience_pack": self.current_experience.to_dict()
        }
    
    def _format_fragment_data(self, frag_type: str, data: Dict) -> Dict:
        """格式化片段数据，确保符合shuchu.json格式"""
        formatted_data = {}
        
        if frag_type == "WHY":
            formatted_data = {
                "goal": data.get("goal", ""),
                "background": data.get("background", ""),
                "constraints": data.get("constraints", []),
                "expected_outcome": data.get("expected_outcome", "")
            }
        elif frag_type == "HOW":
            # 确保HOW片段有steps字段，且格式正确
            steps = data.get("steps", [])
            formatted_steps = []
            
            for step in steps:
                if isinstance(step, dict):
                    formatted_step = {
                        "page": step.get("page", step.get("target", "")),
                        "action": step.get("action", ""),
                        "element": step.get("element", step.get("target", "")),
                        "intent": step.get("intent", step.get("description", ""))
                    }
                    formatted_steps.append(formatted_step)
                else:
                    # 如果是字符串，尝试拆分为动作和元素
                    parts = str(step).split(" ", 1)
                    if len(parts) > 1:
                        formatted_steps.append({
                            "page": "",
                            "action": parts[0],
                            "element": parts[1],
                            "intent": ""
                        })
                    else:
                        formatted_steps.append({
                            "page": "",
                            "action": str(step),
                            "element": "",
                            "intent": ""
                        })
            
            formatted_data["steps"] = formatted_steps
        elif frag_type == "CHECK":
            # 确保CHECK片段有rules字段
            if "rules" in data:
                formatted_data["rules"] = data["rules"]
            else:
                formatted_data["rules"] = [str(rule) for rule in data.values()] if isinstance(data, dict) else [str(data)]
        else:
            # 其他类型直接使用原始数据
            formatted_data = data
        
        return formatted_data
    
    def save_current_experience(self, task_name: str = None, context: str = None) -> Dict:
        """
        保存当前经验
        
        Args:
            task_name: 任务名称
            context: 上下文描述
            
        Returns:
            保存结果
        """
        if not self.current_experience:
            return {
                "success": False,
                "message": "当前没有活动的经验可保存",
                "experience_pack": None
            }
        
        # 更新任务名称
        if task_name:
            self.current_experience.task_name = task_name
        
        # 准备上下文
        if not context:
            # 如果没有提供上下文，使用对话历史或摘要
            if self.recommender.dialogue_history:
                # 仅使用对话历史中的前两轮和后两轮对话作为上下文摘要
                history = self.recommender.dialogue_history
                if len(history) > 4:
                    context_parts = history[:2] + ["..."] + history[-2:]
                    context = "\n".join(context_parts)
                else:
                    context = "\n".join(history)
            else:
                context = f"保存关于「{self.current_experience.task_name}」的经验"
        
        # 保存经验
        result = self.recommender.learn_from_experience(self.current_experience, context)
        
        # 添加到对话历史
        self.recommender.dialogue_history.append(f"系统: 已保存「{self.current_experience.task_name}」经验")
        
        # 添加到上下文历史
        self.context_history.append({
            "role": "system",
            "content": f"保存了「{self.current_experience.task_name}」经验",
            "timestamp": time.time()
        })
        
        experience_dict = self.current_experience.to_dict()
        
        return {
            "success": result["success"],
            "message": result["message"],
            "task_name": self.current_experience.task_name,
            "fragment_count": len(self.current_experience.fragments),
            "experience_pack": experience_dict
        }
    
    def recommend_fragments(self, task_description: str, user_input: str = None) -> Dict:
        """
        推荐经验片段
        
        Args:
            task_description: 任务描述
            user_input: 用户输入上下文
            
        Returns:
            推荐结果
        """
        # 准备上下文
        dialog_context = user_input if user_input else f"寻找与{task_description}相关的经验"
        
        # 添加到对话历史
        if user_input:
            self.recommender.dialogue_history.append(f"用户: {user_input}")
        self.recommender.dialogue_history.append(f"系统: 正在查找与「{task_description}」相关的经验")
        
        # 获取推荐
        existing_fragments = self.current_experience.fragments if self.current_experience else []
        recommendations = self.recommender.recommend_for_task(
            task_description, existing_fragments, dialog_context
        )
        
        # 处理推荐结果，确保符合shuchu.json格式
        result_by_type = {}
        for frag_type, items in recommendations.items():
            if items:
                result_by_type[frag_type] = []
                for item in items:
                    # 格式化内容以符合shuchu.json
                    formatted_content = self._format_fragment_data(
                        frag_type, item["fragment"]["data"]
                    )
                    
                    result_by_type[frag_type].append({
                        "content": formatted_content,
                        "task_name": item["task_name"],
                        "similarity": item["similarity"]
                    })
        
        return {
            "success": True,
            "task_description": task_description,
            "recommendations": result_by_type,
            "has_recommendations": bool(result_by_type),
            "current_experience": self.current_experience.to_dict() if self.current_experience else None
        }
    
    def enhance_experience(self, user_input: str = None) -> Dict:
        """
        增强当前经验
        
        Args:
            user_input: 用户输入上下文
            
        Returns:
            增强建议
        """
        if not self.current_experience:
            return {
                "success": False,
                "message": "当前没有活动的经验可增强",
                "experience_pack": None
            }
        
        # 准备上下文
        dialog_context = user_input if user_input else f"增强「{self.current_experience.task_name}」经验"
        
        # 添加到对话历史
        if user_input:
            self.recommender.dialogue_history.append(f"用户: {user_input}")
        self.recommender.dialogue_history.append(f"系统: 正在分析如何增强当前经验")
        
        # 获取增强建议
        enhance_result = self.recommender.enhance_experience(self.current_experience, dialog_context)
        
        # 处理补充推荐，确保符合shuchu.json格式
        complementary_formatted = {}
        for frag_type, items in enhance_result.get("complementary_recommendations", {}).items():
            if items:
                complementary_formatted[frag_type] = []
                for item in items:
                    # 格式化内容以符合shuchu.json
                    formatted_content = self._format_fragment_data(
                        frag_type, item["fragment"]["data"]
                    )
                    
                    complementary_formatted[frag_type].append({
                        "content": formatted_content,
                        "task_name": item["task_name"],
                        "similarity": item["similarity"]
                    })
        
        return {
            "success": enhance_result["success"],
            "quality_level": enhance_result.get("quality_level", "未知"),
            "missing_types": enhance_result.get("missing_types", []),
            "suggestions": enhance_result.get("enhancement_suggestions", []),
            "has_enhancement_potential": enhance_result.get("has_enhancement_potential", False),
            "complementary_recommendations": complementary_formatted,
            "experience_pack": self.current_experience.to_dict()
        }
    
    def update_workflow_plan(self, steps: List[str]) -> Dict:
        """
        更新工作流计划
        
        Args:
            steps: 工作流步骤列表
            
        Returns:
            更新结果
        """
        if not self.current_experience:
            return {
                "success": False,
                "message": "当前没有活动的经验可更新工作流",
                "experience_pack": None
            }
        
        # 更新工作流计划
        self.current_experience.workflow_plan = {"steps": steps}
        
        # 记录更新
        self.recommender.dialogue_history.append(f"系统: 已更新工作流计划，共{len(steps)}个步骤")
        
        return {
            "success": True,
            "message": f"已更新工作流计划，共{len(steps)}个步骤",
            "workflow_steps": steps,
            "experience_pack": self.current_experience.to_dict()
        }
    
    def get_session_summary(self) -> Dict:
        """
        获取当前会话摘要
        
        Returns:
            会话摘要
        """
        session_duration = time.time() - self.session_start_time
        
        # 统计各类片段数量
        fragment_counts = {}
        if self.current_experience:
            for fragment in self.current_experience.fragments:
                frag_type = fragment.frag_type
                if frag_type not in fragment_counts:
                    fragment_counts[frag_type] = 0
                fragment_counts[frag_type] += 1
        
        return {
            "session_id": str(int(self.session_start_time)),
            "task": self.current_task or "未命名任务",
            "duration_seconds": int(session_duration),
            "context_history_length": len(self.context_history),
            "dialogue_history_length": len(self.recommender.dialogue_history),
            "current_experience": self.current_experience.to_dict() if self.current_experience else None
        }
    def save_to_file(self, file_path: str = "shuchu1.json") -> Dict:
        """
        将当前经验包保存到指定的JSON文件

        Args:
            file_path: 保存的文件路径，默认为shuchu.json
            
        Returns:
            保存结果
        """
        if not self.current_experience:
            return {
                "success": False,
                "message": "当前没有活动的经验可保存",
                "file_path": None
            }

        try:
            # 获取完整的经验包数据
            experience_dict = self.current_experience.to_dict()
            
            # 写入文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(experience_dict, f, ensure_ascii=False, indent=2)
            
            return {
                "success": True,
                "message": f"已将经验包保存到 {file_path}",
                "file_path": file_path,
                "experience_pack": experience_dict
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"保存到文件失败: {str(e)}",
                "file_path": file_path,
                "error": str(e)
            }
    
    def save_session(self) -> Dict:
        """
        保存当前会话
        
        Returns:
            保存结果
        """
        # 保存当前经验
        if self.current_experience and self.current_experience.fragments:
            # 保存到数据库
            save_result = self.save_current_experience()
            experience_saved = save_result["success"]
            experience_pack = save_result.get("experience_pack")
            
            # 额外保存到shuchu.json
            save_file_result = self.save_to_file()
            file_saved = save_file_result["success"]
        else:
            experience_saved = False
            experience_pack = None
            file_saved = False
        
        # 保存对话历史
        if self.recommender.dialogue_history:
            dialogue_saved = self.recommender.save_dialogue_history()
        else:
            dialogue_saved = False
        
        return {
            "success": experience_saved or dialogue_saved or file_saved,
            "experience_saved": experience_saved,
            "dialogue_saved": dialogue_saved,
            "file_saved": file_saved,
            "message": "会话已保存",
            "session_summary": self.get_session_summary(),
            "experience_pack": experience_pack
        }
    
    def export_experience(self, format_type: str = "json") -> Dict:
        """
        导出当前经验
        
        Args:
            format_type: 导出格式 (json, dict)
            
        Returns:
            导出的经验
        """
        if not self.current_experience:
            return {
                "success": False,
                "message": "当前没有活动的经验可导出",
                "data": None
            }
        
        # 转换为标准shuchu.json格式
        experience_dict = self.current_experience.to_dict()
        
        if format_type == "json":
            try:
                data = json.dumps(experience_dict, ensure_ascii=False, indent=2)
            except Exception as e:
                return {
                    "success": False,
                    "message": f"导出JSON格式失败: {str(e)}",
                    "data": None
                }
        else:
            data = experience_dict
        
        return {
            "success": True,
            "message": f"已导出{format_type}格式经验",
            "data": data,
            "experience_pack": experience_dict
        }


# 示例用法
if __name__ == "__main__":
    # 初始化智能体控制器
    agent = ControllerAgent("rich_expert_validation.json")
    
    # 开始新会话
    session = agent.new_session("网页自动化测试项目")
    print(f"会话开始: {session['message']}")
    
    # 添加WHY片段
    agent.add_fragment("WHY", {
        "goal": "建立网页界面自动验证系统",
        "background": "在大促活动中页面频繁变化需要快速验证",
        "constraints": ["必须基于现有HTML结构", "要有审计日志功能"],
        "expected_outcome": "能自动检测并报告页面关键元素变化"
    })
    
    # 添加HOW片段
    agent.add_fragment("HOW", {
        "steps": [
            {"page": "元素抽取页", "action": "点击", "element": "结构化提取按钮", "intent": "提取页面HTML中的元素信息"},
            {"page": "验证页", "action": "填写", "element": "测试数据输入框", "intent": "填写拟合后的页面参数进行验证"},
            {"page": "验证页", "action": "点击", "element": "开始验证按钮", "intent": "执行自动回放流程"}
        ]
    })
    
    # 添加CHECK片段
    agent.add_fragment("CHECK", {
        "rules": [
            "每个页面元素必须具备唯一定位属性",
            "页面的结构变化必须在统计配置项中显式声明",
            "执行路径需含验证步骤，输出结果包含status/msg字段"
        ]
    })
    
    # 导出经验为JSON
    result = agent.export_experience()
    print("\n导出的经验包:")
    print(result["data"])
    
    # 处理用户输入
    agent.process_user_input("我需要一个系统来验证网页界面的变化")
    
    # 保存会话
    save_result = agent.save_session()
    print(f"\n保存会话结果: {save_result['message']}")