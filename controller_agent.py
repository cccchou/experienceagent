# experienceagent/controller_agent.py
"""
控制器智能体
负责整个经验学习流程的调度和执行
"""

from typing import Dict, List, Any, Literal
import logging
from experienceagent.agents.initiation_agent import InitiationAgent
from experienceagent.agents.observation_agent import ObservationAgent
from experienceagent.agents.extraction_agent import ExtractionAgent
from experienceagent.agents.fusion_agent import FusionAgent
from experienceagent.fragment_scorer import ExperienceEvaluator
from experienceagent.fragment_recommender import ExperienceRetriever
from experienceagent.knowledge_graph import ExperienceGraph

# 导入基础控制器
from agents.controller_agent import ControllerAgent as BaseController

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ControllerAgent")

# 从基础控制器继承Stage类型
Stage = Literal[
    "initiation", "observation", "extraction", "fusion", "practice"
]

class ControllerAgent(BaseController):
    """
    经验学习流程控制器
    继承自基础控制器，增加完整经验学习流程支持
    """
    def __init__(self, config: Dict = None):
        # 初始化基础控制器
        super().__init__()
        self.config = config or {}
        
        # 创建知识图谱
        self.knowledge_graph = ExperienceGraph("main_graph")
        
        # 初始化子阶段智能体
        self.initiation_agent = InitiationAgent()
        self.observation_agent = ObservationAgent()
        self.extraction_agent = ExtractionAgent(graph=self.knowledge_graph)
        self.fusion_agent = FusionAgent(graph=self.knowledge_graph)
        
        # 初始化评估器和检索器
        self.evaluator = ExperienceEvaluator()
        experience_db_path = self.config.get("experience_db_path")
        self.retriever = ExperienceRetriever(experience_db_path)
        
        # 运行时数据
        self.input_data = None
        self.experience_pack = None
        self.execution_log = []
        self.recommendations = {}
    
    def run_full_pipeline(self, input_data=None):
        """
        执行完整的经验学习流程
        
        Args:
            input_data: 输入数据对象
            
        Returns:
            执行结果摘要
        """
        self.input_data = input_data
        if not self.input_data:
            logger.error("未提供输入数据")
            return {"status": "failed", "error": "未提供输入数据", "final_stage": self.current_stage}
            
        try:
            # 记录开始状态
            logger.info("开始执行经验学习流程")
            self._log_execution("start", {"stage": self.current_stage})
            
            # 初始化阶段
            self._execute_initiation_stage()
            
            # 按控制器状态执行后续阶段
            while self.current_stage != "practice":
                if self.current_stage == "observation":
                    self._execute_observation_stage()
                elif self.current_stage == "extraction":
                    self._execute_extraction_stage()
                elif self.current_stage == "fusion":
                    self._execute_fusion_stage()
                else:
                    break  # 不应该到达这里，但为安全起见
                
            # 如果执行到实践阶段，则执行
            if self.current_stage == "practice":
                self._execute_practice_stage()
                
            # 评估经验包质量
            if self.experience_pack:
                evaluation = self.evaluator.evaluate_experience_pack(self.experience_pack)
                self.experience_pack.trust_score = evaluation.get("overall_score", 0.5)
                
                # 记录评估结果
                self._log_execution("evaluation", {
                    "overall_score": evaluation.get("overall_score", 0),
                    "suggestions": evaluation.get("suggestions", [])
                })
            
            # 返回执行摘要
            return {
                "status": "success",
                "task_name": getattr(self.experience_pack, "task_name", "未命名任务"),
                "final_stage": self.current_stage,
                "trust_score": getattr(self.experience_pack, "trust_score", 0),
                "stage_history": self.stage_history
            }
                
        except Exception as e:
            logger.error(f"执行过程中发生错误: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "status": "error",
                "error": str(e),
                "stage": self.current_stage,
                "final_stage": self.current_stage  # 确保错误情况下也有final_stage键
            }
    
    def _execute_initiation_stage(self):
        """执行初始化阶段"""
        logger.info("执行初始化阶段")
        
        # 改用正确的方法名 - 这里我们假设InitiationAgent有一个process方法
        # 如果实际方法名不同，请根据InitiationAgent的实际接口修改
        if hasattr(self.initiation_agent, "process"):
            task_info = self.initiation_agent.process(self.input_data)
        else:
            # 如果没有明确的方法接口，我们创建一个默认的任务信息
            logger.warning("InitiationAgent接口不匹配，使用默认任务信息")
            task_info = {"task_name": "未命名任务"}
        
        # 创建经验包
        from experienceagent.goalfylearning import ExperiencePack, WhyBuilder
        task_name = task_info.get("task_name", "未命名任务")
        self.experience_pack = ExperiencePack(task_name)
        
        # 创建WHY片段
        if hasattr(self.input_data, "interview_logs") and self.input_data.interview_logs:
            why_fragment = WhyBuilder.from_interview(self.input_data.interview_logs)
            self.experience_pack.add_fragment(why_fragment)
        
        # 更新控制器上下文，进入下一阶段
        ctx = {"confirmed": True, "valuable": True}
        next_stage = self.update_context(ctx)
        
        # 记录执行
        self._log_execution("initiation", {
            "task_name": task_name,
            "next_stage": next_stage
        })
    
    def _execute_observation_stage(self):
        """执行观察阶段"""
        logger.info("执行观察阶段")
        
        # 检查是否有行为日志
        has_logs = hasattr(self.input_data, "behavior_logs") and self.input_data.behavior_logs
        
        if not has_logs:
            logger.info("无行为日志，跳过观察阶段")
            ctx = {"logs_complete": True, "within_scope": True}
            self.update_context(ctx)
            return
        
        # 调用观察智能体 - 确保方法名称正确
        observations = None
        if hasattr(self.observation_agent, "observe"):
            observations = self.observation_agent.observe(self.input_data)
        else:
            logger.warning("ObservationAgent接口不匹配")
        
        # 创建HOW片段
        if observations:
            from experienceagent.goalfylearning import ExperienceFragment
            how_fragment = ExperienceFragment("HOW")
            how_fragment.data = {"steps": observations}
            self.experience_pack.add_fragment(how_fragment)
        
        # 更新上下文
        ctx = {
            "logs_complete": observations is not None,
            "within_scope": True
        }
        next_stage = self.update_context(ctx)
        
        # 记录执行
        self._log_execution("observation", {
            "observations_count": len(observations) if observations else 0,
            "next_stage": next_stage
        })
    
    def _execute_extraction_stage(self):
        """执行提取阶段"""
        logger.info("执行提取阶段")
        
        # 调用提取智能体 - 确保方法名称正确
        knowledge_points = None
        if hasattr(self.extraction_agent, "extract"):
            knowledge_points = self.extraction_agent.extract(self.input_data)
        else:
            logger.warning("ExtractionAgent接口不匹配")
        
        # 将知识点添加到经验图谱
        if knowledge_points:
            for kp in knowledge_points:
                self.experience_pack.kg.add_node(kp)
        
        # 更新上下文
        ctx = {"subjective_complete": True}
        next_stage = self.update_context(ctx)
        
        # 记录执行
        self._log_execution("extraction", {
            "knowledge_points_count": len(knowledge_points) if knowledge_points else 0,
            "next_stage": next_stage
        })
    
    def _execute_fusion_stage(self):
        """执行融合阶段"""
        logger.info("执行融合阶段")
        
        # 检索相似经验
        similar_experiences = []
        if hasattr(self.experience_pack, "task_name"):
            try:
                results = self.retriever.retrieve_by_task_name(self.experience_pack.task_name)
                similar_experiences = [r["experience"] for r in results]
            except Exception as e:
                logger.error(f"检索相似经验失败: {str(e)}")
        
        # 调用融合智能体 - 确保方法名称正确
        fusion_result = {"success": False}
        if hasattr(self.fusion_agent, "fuse"):
            fusion_input = {
                "experience_pack": self.experience_pack,
                "similar_experiences": similar_experiences
            }
            try:
                fusion_result = self.fusion_agent.fuse(fusion_input)
            except Exception as e:
                logger.error(f"融合失败: {str(e)}")
        else:
            logger.warning("FusionAgent接口不匹配")
        
        # 推荐校验规则
        self._recommend_check_rules()
        
        # 更新上下文
        ctx = {"fusion_success": fusion_result.get("success", True)}
        next_stage = self.update_context(ctx)
        
        # 记录执行
        self._log_execution("fusion", {
            "fusion_success": fusion_result.get("success", True),
            "similar_experiences_count": len(similar_experiences),
            "next_stage": next_stage
        })
    
    def _execute_practice_stage(self):
        """执行实践阶段"""
        logger.info("执行实践阶段")
        
        # 构建工作流
        from experienceagent.goalfylearning import WorkflowBuilder
        workflow = WorkflowBuilder.build_from_fragments(self.experience_pack.fragments)
        
        # 更新上下文
        ctx = {"errors": False}  # 假设没有错误
        self.update_context(ctx)
        
        # 记录执行
        self._log_execution("practice", {
            "workflow_steps": len(workflow.get("steps", [])),
        })
    
    def _recommend_check_rules(self):
        """推荐CHECK规则"""
        # 检索CHECK类型片段
        check_fragments = []
        try:
            check_fragments = self.retriever.retrieve_by_fragment_type("CHECK")
        except Exception as e:
            logger.error(f"检索CHECK片段失败: {str(e)}")
        
        if check_fragments:
            # 简单实现：选择第一个
            check_frag_data = check_fragments[0]["fragment"].get("data", {})
            rules = check_frag_data.get("rules", [])
            
            if rules:
                from experienceagent.goalfylearning import CheckBuilder
                check_fragment = CheckBuilder.from_rules(rules)
                self.experience_pack.add_fragment(check_fragment)
                logger.info(f"已添加推荐的校验规则，共 {len(rules)} 条")
                return
        
        # 如果没有找到合适的规则，创建空规则
        from experienceagent.goalfylearning import CheckBuilder
        check_fragment = CheckBuilder.from_rules([])
        self.experience_pack.add_fragment(check_fragment)
        logger.info("未找到合适校验规则，已添加空规则集")
    
    def _log_execution(self, stage: str, data: Dict):
        """记录执行历史"""
        import time
        self.execution_log.append({
            "stage": stage,
            "timestamp": time.time(),
            "data": data
        })
    
    def get_experience_summary(self):
        """获取经验包摘要"""
        if not self.experience_pack:
            return {"status": "未创建经验包"}
            
        return {
            "task_name": self.experience_pack.task_name,
            "trust_score": self.experience_pack.trust_score,
            "fragments": [f.frag_type for f in self.experience_pack.fragments],
            "knowledge_points": len(self.experience_pack.kg.get_all_nodes()) if hasattr(self.experience_pack.kg, "get_all_nodes") else 0
        }
    
    def save_experience(self, path: str = None):
        """
        保存经验到数据库
        
        Args:
            path: 保存路径（可选）
            
        Returns:
            是否保存成功
        """
        if not self.experience_pack:
            logger.error("没有可保存的经验")
            return False
            
        # 添加到检索器
        self.retriever.add_experience(self.experience_pack)
        
        # 保存到文件
        save_path = path or self.config.get("experience_db_path")
        if save_path:
            return self.retriever.save_db(save_path)
        
        return True  # 仅保存在内存中


# 示例用法
if __name__ == "__main__":
    # 测试控制器
    controller = ControllerAgent()
    
    # 模拟输入数据
    class MockInput:
        def __init__(self):
            self.interview_logs = [
                "我需要一个自动验证页面的工具",
                "用于检测页面变化并生成报告"
            ]
            self.behavior_logs = [
                {"action": "click", "element": "button", "time": 1623456789}
            ]
    
    # 执行流程
    result = controller.run_full_pipeline(MockInput())
    print(f"执行结果: {result['status']}")
    
    # 安全访问字典键
    final_stage = result.get('final_stage', '未知')
    print(f"最终阶段: {final_stage}")