# experienceagent/controller_agent.py
"""
经验智能体控制器
负责协调经验检索、推荐和学习，匹配shuchu.json格式
支持GPT自动生成功能，当经验库无匹配时自动补充
"""

import json
import logging
from typing import List, Dict, Optional
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
        # self.version = 1
        # self.trust_score = 0.5  # 默认信任分数
        self.workflow_plan = {"steps": []}  # 添加workflow_plan字段
    
    def add_fragment(self, fragment: ExperienceFragment):
        """添加经验片段"""
        self.fragments.append(fragment)
        if fragment.frag_type == "WHY":
            self.trust_score = fragment.data.get('similarity', '')  # 更新信任分数
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
            "trust_score": self.trust_score,
            "fragments": [f.to_dict() for f in self.fragments],
            "workflow_plan": self.workflow_plan
        }


class ControllerAgent:
    """
    经验智能体控制器
    协调经验检索、推荐和学习
    """
    
    def __init__(self, db_path: str = "rich_expert_validation.json", min_recommendations: int = 2, 
                min_similarity: float = 0.89):
        """
        初始化智能体控制器
        
        Args:
            db_path: 经验数据库路径
            min_recommendations: 每种类型的最少推荐数量
            min_similarity: 最低相似度阈值
        """
        self.db_path = db_path
        
        # 初始化评估器
        self.evaluator = ExperienceEvaluator()
        
        # 初始化检索器和推荐器 - 设置GPT生成阈值
        self.retriever = ExperienceRetriever(db_path, self.evaluator)
        self.recommender = FragmentRecommender(
            self.retriever, 
            self.evaluator,
            min_recommendations=min_recommendations,
            min_similarity=min_similarity
        )
        
        # 会话状态
        self.current_task = None
        
    
    def new_session(self, task_name: str = None) -> Dict:
        """
        开始新的会话
        
        Args:
            task_name: 任务名称,主要获得的是凝练出来的task_name
        
        Returns:
            会话信息
        """
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
        }
    
    def run(self, user_input: list) -> Dict:
        """
        处理用户输入
        
        Args:
            user_input: 用户输入
            task_hint: 任务提示
            
        Returns:
            处理结果
        """
        
        # 根据用户输入寻找相关经验
        search_query = self.current_task
        result = self.recommender.recommend_for_task(search_query, user_input)
        return result
        
    
    def add_fragments(self, fragment_data: Dict) -> Dict:
        """
        添加经验片段
        为了保存到json格式
        
        Args:
            frag_type: 片段类型 (WHY, HOW, CHECK)
            fragment_data: 片段数据
            user_input: 用户输入上下文
            
        Returns:
            添加结果
        """
        # 确保有当前经验
        self.current_experience = ExperiencePack(self.current_task)
        
        # 格式化片段数据，确保符合shuchu.json格式
        for i in ["WHY",'HOW','CHECK']:
            formatted_data = self._format_fragment_data(i, fragment_data[i])
        
        # 创建并添加片段
            fragment = ExperienceFragment(i, formatted_data)
            self.current_experience.add_fragment(fragment)
        
        
        return {
            "success": True,
            "fragment_count": len(self.current_experience.fragments),
            "experience_pack": self.current_experience.to_dict()
        }
    
    def _format_fragment_data(self, frag_type: str, data: Dict) -> Dict:
        """格式化片段数据，确保符合shuchu.json格式"""
        '选第一个相似度最高'
        temp = data[0]['fragment'].get("data", {})
        formatted_data = {}
        if frag_type == "WHY":
            formatted_data = {
                "goal": temp.get("goal", ""),
                "background": temp.get("background", ""),
                "constraints": temp.get("constraints", []),
                "expected_outcome": temp.get("expected_outcome", ""),
                "similarity": data[0].get("similarity", ''),
            }
        elif frag_type == "HOW":
            # 确保HOW片段有steps字段，且格式正确
            steps = temp.get("steps", [])
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
            if "rules" in temp:
                formatted_data["rules"] = temp["rules"]
        else:
            # 其他类型直接使用原始数据
            logger.warning(f"未知片段类型: {frag_type}, 请检查")
            return 1
        return formatted_data
    
    def save_current_experience(self, task_name: str = None) -> Dict:
        """
        保存当前经验
        记录用
        Args:
            task_name: 任务名称
            context: 上下文描述
            
        Returns:
            保存结果
        """
        
        
        # 更新任务名称
        if task_name:
            self.current_experience.task_name = task_name
        
        # 保存经验
        experience_dict = self.current_experience.to_dict()
        
        return {
            "task_name": self.current_experience.task_name,
            "fragment_count": len(self.current_experience.fragments),
            "experience_pack": experience_dict
        }
    
    def get_session_summary(self) -> Dict:
        """
        获取当前会话摘要
        
        Returns:
            会话摘要
        """
        # session_duration = time.time() - self.session_start_time
        
        # 统计各类片段数量
        fragment_counts = {}
        if self.current_experience:
            for fragment in self.current_experience.fragments:
                frag_type = fragment.frag_type
                if frag_type not in fragment_counts:
                    fragment_counts[frag_type] = 0
                fragment_counts[frag_type] += 1
        
        return {
            "task": self.current_task or "未命名任务",
            "dialogue_history_length": len(self.recommender.dialogue_history),
            "current_experience": self.current_experience.to_dict() if self.current_experience else None
        }
    
    def save_session(self,data:dict) -> Dict:
        """
        保存当前会话
        
        Returns:
            只保存结果，没保存到文件
        """
        self.add_fragments(data)
        # 保存当前经验
        if self.current_experience:
            save_result = self.save_current_experience()
            experience_pack = save_result.get("experience_pack")
        else:
            experience_pack = None
        
        # 保存对话历史
        if self.recommender.dialogue_history:
            #dialogue history只保存到经验体，输出json里不包含
            dialogue_saved = self.recommender.save_dialogue_history()
        else:
            dialogue_saved = False
        
        return {
            "dialogue_saved": dialogue_saved,
            "message": "会话已保存",
            "session_summary": self.get_session_summary(),
            "experience_pack": experience_pack
        }
    
    def save_to_file(self, file_path: str = "shuchu.json") -> Dict:
        """
        将当前经验包保存到指定的JSON文件，完全遵循模板格式
        
        Args:
            file_path: 保存的文件路径，默认为shuchu.json
            
        Returns:
            保存结果
        """
        try:
            # 获取完整的经验包数据
            experience_dict = self.current_experience.to_dict()
            
            # 确保数据结构完整且符合模板
            if "fragments" not in experience_dict:
                experience_dict["fragments"] = []
            
            if "workflow_plan" not in experience_dict:
                experience_dict["workflow_plan"] = {"steps": []}
            elif "steps" not in experience_dict["workflow_plan"]:
                experience_dict["workflow_plan"]["steps"] = []
            
            if "trust_score" not in experience_dict:
                experience_dict["trust_score"] = 0.5
            
            # 写入文件，使用缩进格式确保可读性
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
        
    # 配置项
    @property
    def auto_adopt_ai_recommendations(self) -> bool:
        """是否自动采纳AI推荐"""
        return True  # 默认启用


# 示例用法
if __name__ == "__main__":
    db_path = "rich_expert_validation.json"
    evaluator = ExperienceEvaluator()
    retriever = ExperienceRetriever(db_path, evaluator)
    recommender = FragmentRecommender(retriever, evaluator)
    # 初始化智能体控制器
    agent = ControllerAgent(db_path=db_path)
    # 开始新会话,task和dialoge——hsitory是需要goalfylearning处理输入
    session = agent.new_session("网页自动化测试项目")
    print(f"会话开始: {session['message']}")
    #goalfylearning处理输入
    dialogue_hsitory = ['我想完成一个网页自动化测试的插件']
    result = agent.run(dialogue_hsitory)
    # # 保存会话
    save_result = agent.save_session(data = result)
    print(f"\n保存会话结果: {save_result['message']}")
    # 保存到shuchu.json
    agent.save_to_file()
    print("已保存到shuchu.json")
    