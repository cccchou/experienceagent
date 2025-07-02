# experienceagent/controller_agent.py
"""
经验智能体控制器
负责协调经验检索、推荐和学习，匹配shuchu.json格式
支持GPT自动生成功能，当经验库无匹配时自动补充
"""

from collections import defaultdict
import json
import logging
from typing import List, Dict, Optional
from experienceagent.fragment_recommender import ExperienceRetriever, FragmentRecommender
from experienceagent.fragment_scorer import ExperienceEvaluator
from experienceagent.knowledage import (
    KnowledgeGraph, 
    KnowledgeNode,
    build_knowledge_graph_from_experiences,
    save_knowledge_graph,
    load_knowledge_graph,
    extract_knowledge_from_experience
)


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
        self.knowledge_graph = None
        self.kg_path = db_path.replace('.json', '_knowledge_graph.json')
        
        # 尝试加载现有知识图谱
        self._load_knowledge_graph()
    
    def _load_knowledge_graph(self):
        """加载知识图谱"""
        try:
            self.knowledge_graph = load_knowledge_graph(self.kg_path)
            if self.knowledge_graph:
                logger.info("知识图谱加载成功")
            else:
                logger.info("未找到现有知识图谱，将在需要时创建")
        except Exception as e:
            logger.warning(f"加载知识图谱失败: {str(e)}")
            self.knowledge_graph = None
    
    def enhance_knowledge(self, force_rebuild: bool = False, target_experience_id: str = None) -> Dict:
        """
        对最新的经验体进行关联成知识图谱
        
        Args:
            force_rebuild: 是否强制重建整个知识图谱
            target_experience_id: 指定处理的经验体ID，如果为None则处理所有经验体
            
        Returns:
            操作结果
        """
        try:
            logger.info("开始知识图谱增强...")
            
            # 获取经验库数据
            if not hasattr(self, 'retriever') or not self.retriever.experience_db:
                return {
                    "success": False,
                    "message": "经验库为空或未加载",
                    "details": {}
                }
            
            experiences = self.retriever.experience_db
            
            # 如果强制重建或没有现有图谱，则重建
            if force_rebuild or not self.knowledge_graph:
                logger.info("构建新的知识图谱...")
                self.knowledge_graph = build_knowledge_graph_from_experiences(experiences)
                operation = "rebuild"
                processed_count = len(experiences)
            else:
                # 增量更新模式
                logger.info("增量更新知识图谱...")
                processed_count = 0
                
                if target_experience_id:
                    # 处理指定经验体
                    target_exp = None
                    for i, exp in enumerate(experiences):
                        exp_id = f"experience_{i+1:03d}"
                        if exp_id == target_experience_id:
                            target_exp = exp
                            break
                    
                    if target_exp:
                        # 删除旧的知识节点（如果存在）
                        self._remove_experience_knowledge(target_experience_id)
                        
                        # 添加新的知识节点
                        new_nodes = extract_knowledge_from_experience(target_exp, target_experience_id)
                        for node in new_nodes:
                            self.knowledge_graph.add_knowledge_node(node)
                        processed_count = 1
                        operation = "incremental_update_single"
                    else:
                        return {
                            "success": False,
                            "message": f"未找到指定的经验体: {target_experience_id}",
                            "details": {}
                        }
                else:
                    # 检查是否有新的经验体需要处理
                    existing_exp_ids = set()
                    for node in self.knowledge_graph.knowledge_nodes.values():
                        if node.experience_id:
                            existing_exp_ids.add(node.experience_id)
                    
                    # 处理新增的经验体
                    for i, exp in enumerate(experiences):
                        exp_id = f"experience_{i+1:03d}"
                        if exp_id not in existing_exp_ids:
                            new_nodes = extract_knowledge_from_experience(exp, exp_id)
                            for node in new_nodes:
                                self.knowledge_graph.add_knowledge_node(node)
                            processed_count += 1
                    
                    operation = "incremental_update_batch" if processed_count > 0 else "no_update_needed"
            
            # 保存知识图谱
            save_success = save_knowledge_graph(self.knowledge_graph, self.kg_path)
            
            # 获取统计信息
            stats = self.knowledge_graph.get_statistics()
            stats["operation"] = operation
            stats["processed_experiences"] = processed_count
            
            # 分析关联关系
            relationship_analysis = self._analyze_knowledge_relationships()
            
            result = {
                "success": True,
                "message": f"知识图谱增强完成 ({operation})",
                "stats": stats,
                "relationships": relationship_analysis,
                "saved": save_success,
                "kg_path": self.kg_path if save_success else None
            }
            
            logger.info(f"知识图谱增强完成: 处理了 {processed_count} 个经验体")
            return result
            
        except Exception as e:
            error_msg = f"知识图谱增强失败: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "message": error_msg,
                "details": {"error": str(e)}
            }
    
    def _remove_experience_knowledge(self, experience_id: str):
        """移除指定经验体的知识节点"""
        try:
            nodes_to_remove = []
            for kid, node in self.knowledge_graph.knowledge_nodes.items():
                if node.experience_id == experience_id:
                    nodes_to_remove.append(kid)
            
            for kid in nodes_to_remove:
                del self.knowledge_graph.knowledge_nodes[kid]
                # 同时需要更新索引，这里简化处理，实际可能需要重建索引
            
            # 重建索引
            self.knowledge_graph = KnowledgeGraph.from_dict(self.knowledge_graph.to_dict())
            
            logger.info(f"移除经验体 {experience_id} 的 {len(nodes_to_remove)} 个知识节点")
            
        except Exception as e:
            logger.error(f"移除经验体知识节点失败: {str(e)}")
    
    def _analyze_knowledge_relationships(self) -> Dict:
        """分析知识关联关系"""
        try:
            analysis = {
                "cross_experience_connections": 0,
                "system_clusters": {},
                "page_clusters": {},
                "domain_clusters": {},
                "isolated_nodes": 0
            }
            
            # 统计跨经验体连接
            experience_systems = defaultdict(set)
            experience_pages = defaultdict(set)
            experience_domains = defaultdict(set)
            
            for node in self.knowledge_graph.knowledge_nodes.values():
                if node.experience_id:
                    experience_systems[node.experience_id].update(node.system_ids)
                    experience_pages[node.experience_id].update(node.page_ids)
                    experience_domains[node.experience_id].update(node.domain_ids)
            
            # 分析系统集群
            for system_id in self.knowledge_graph.system_index:
                related_exps = set()
                for node_id in self.knowledge_graph.system_index[system_id]:
                    node = self.knowledge_graph.knowledge_nodes[node_id]
                    if node.experience_id:
                        related_exps.add(node.experience_id)
                
                if len(related_exps) > 1:
                    analysis["system_clusters"][system_id] = {
                        "experience_count": len(related_exps),
                        "knowledge_count": len(self.knowledge_graph.system_index[system_id]),
                        "experiences": list(related_exps)
                    }
                    analysis["cross_experience_connections"] += len(related_exps) - 1
            
            # 分析页面集群
            for page_id in self.knowledge_graph.page_index:
                related_exps = set()
                for node_id in self.knowledge_graph.page_index[page_id]:
                    node = self.knowledge_graph.knowledge_nodes[node_id]
                    if node.experience_id:
                        related_exps.add(node.experience_id)
                
                if len(related_exps) > 1:
                    analysis["page_clusters"][page_id] = {
                        "experience_count": len(related_exps),
                        "knowledge_count": len(self.knowledge_graph.page_index[page_id]),
                        "experiences": list(related_exps)
                    }
            
            # 分析领域集群
            for domain_id in self.knowledge_graph.domain_index:
                related_exps = set()
                for node_id in self.knowledge_graph.domain_index[domain_id]:
                    node = self.knowledge_graph.knowledge_nodes[node_id]
                    if node.experience_id:
                        related_exps.add(node.experience_id)
                
                if len(related_exps) > 1:
                    analysis["domain_clusters"][domain_id] = {
                        "experience_count": len(related_exps),
                        "knowledge_count": len(self.knowledge_graph.domain_index[domain_id]),
                        "experiences": list(related_exps)
                    }
            
            # 统计孤立节点
            for node in self.knowledge_graph.knowledge_nodes.values():
                if (not node.system_ids and not node.page_ids and not node.domain_ids):
                    analysis["isolated_nodes"] += 1
            
            return analysis
            
        except Exception as e:
            logger.error(f"关联关系分析失败: {str(e)}")
            return {}
    
    def query_knowledge_graph(self, query: str, query_type: str = "content", 
                            filters: Dict = None) -> Dict:
        """
        查询知识图谱
        
        Args:
            query: 查询内容
            query_type: 查询类型 (content/experience/system/page/domain/related)
            filters: 过滤条件
            
        Returns:
            查询结果
        """
        if not self.knowledge_graph:
            return {
                "success": False,
                "message": "知识图谱未初始化",
                "results": []
            }
        
        try:
            results = []
            
            if query_type == "content":
                # 内容搜索
                nodes = self.knowledge_graph.search_knowledge_by_content(query)
                results = [node.to_dict() for node in nodes]
                
            elif query_type == "experience":
                # 按经验体查询
                nodes = self.knowledge_graph.get_knowledge_by_experience(query)
                results = [node.to_dict() for node in nodes]
                
            elif query_type == "system":
                # 按系统查询
                nodes = self.knowledge_graph.get_knowledge_by_system(query)
                results = [node.to_dict() for node in nodes]
                
            elif query_type == "page":
                # 按页面查询
                nodes = self.knowledge_graph.get_knowledge_by_page(query)
                results = [node.to_dict() for node in nodes]
                
            elif query_type == "domain":
                # 按领域查询
                nodes = self.knowledge_graph.get_knowledge_by_domain(query)
                results = [node.to_dict() for node in nodes]
                
            elif query_type == "related":
                # 相关知识查询
                related_dict = self.knowledge_graph.find_related_knowledge(query)
                results = {
                    relation_type: [node.to_dict() for node in nodes]
                    for relation_type, nodes in related_dict.items()
                }
            
            # 应用过滤器
            if filters and isinstance(results, list):
                results = self._apply_filters(results, filters)
            
            return {
                "success": True,
                "message": f"找到 {len(results)} 个相关结果",
                "query_type": query_type,
                "results": results
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"查询失败: {str(e)}",
                "results": []
            }
    
    def _apply_filters(self, results: List[Dict], filters: Dict) -> List[Dict]:
        """应用查询过滤器"""
        filtered_results = results
        
        if "experience_id" in filters:
            filtered_results = [r for r in filtered_results 
                             if r.get("experience_id") == filters["experience_id"]]
        
        if "system_ids" in filters:
            target_systems = set(filters["system_ids"])
            filtered_results = [r for r in filtered_results 
                             if target_systems.intersection(set(r.get("system_ids", [])))]
        
        if "domain_ids" in filters:
            target_domains = set(filters["domain_ids"])
            filtered_results = [r for r in filtered_results 
                             if target_domains.intersection(set(r.get("domain_ids", [])))]
        
        return filtered_results
    
    def get_knowledge_stats(self) -> Dict:
        """获取知识图谱统计信息"""
        if not self.knowledge_graph:
            return {"message": "知识图谱未初始化"}
        
        stats = self.knowledge_graph.get_statistics()
        
        # 添加详细统计
        stats["relationship_analysis"] = self._analyze_knowledge_relationships()
        
        return stats
        
    
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
    # db_path = "rich_expert_validation.json"
    # evaluator = ExperienceEvaluator()
    # retriever = ExperienceRetriever(db_path, evaluator)
    # recommender = FragmentRecommender(retriever, evaluator)
    # # 初始化智能体控制器
    # agent = ControllerAgent(db_path=db_path)
    # # 开始新会话,task和dialoge——hsitory是需要goalfylearning处理输入
    # session = agent.new_session("网页自动化测试项目")
    # print(f"会话开始: {session['message']}")
    # #goalfylearning处理输入
    # dialogue_hsitory = ['我想完成一个网页自动化测试的插件']
    # result = agent.run(dialogue_hsitory)
    # # # 保存会话
    # save_result = agent.save_session(data = result)
    # print(f"\n保存会话结果: {save_result['message']}")
    # # 保存到shuchu.json
    # agent.save_to_file()
    # print("已保存到shuchu.json")

    # # 查询知识图谱
    # 初始化智能体
    agent = ControllerAgent(db_path="rich_expert_validation.json")

    # 构建/更新知识图谱
    result = agent.enhance_knowledge(force_rebuild=True)
    print(f"知识图谱构建结果: {result}")

    # 查询特定经验体的知识
    query_result = agent.query_knowledge_graph("experience_001", query_type="experience")
    print(f"经验体知识: {query_result}")

    # 按系统查询
    system_result = agent.query_knowledge_graph("example.com", query_type="system")
    print(f"系统相关知识: {system_result}")

    # 按内容搜索
    content_result = agent.query_knowledge_graph("页面验证", query_type="content")
    print(f"内容搜索结果: {content_result}")

    # 获取统计信息
    stats = agent.get_knowledge_stats()
    print(f"知识图谱统计: {stats}")
    