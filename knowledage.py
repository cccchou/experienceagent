"""
知识图谱模块
用于构建和管理经验体的知识图谱，实现知识点的关联和检索
基于指定的图谱字段样例实现
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class KnowledgeNode:
    """知识节点类 - l_knowledge节点类型"""
    
    def __init__(self, knowledge_id: str = None, experience_id: str = None,
                 extracted_subjective_knowledge_id: str = None, operation_steps_id: str = None,
                 system_ids: List[str] = None, page_ids: List[str] = None, 
                 domain_ids: List[str] = None, created_at: str = None, updated_at: str = None):
        """
        初始化知识节点
        
        Args:
            knowledge_id: 知识节点的唯一标识符
            experience_id: 经验体ID，关联该主观知识属于哪个经验体
            extracted_subjective_knowledge_id: 挖掘知识ID，若有的话，仅有一个，可选
            operation_steps_id: 操作步骤ID，若有的话，仅有一个，可选
            system_ids: 关联的系统ID列表，支持多个
            page_ids: 关联的页面ID列表，支持多个
            domain_ids: 关联的业务领域ID列表，支持多个
            created_at: 创建时间
            updated_at: 更新时间
        """
        self.knowledge_id = knowledge_id or f"kn{str(uuid.uuid4())[:8]}"
        self.experience_id = experience_id
        self.extracted_subjective_knowledge_id = extracted_subjective_knowledge_id
        self.operation_steps_id = operation_steps_id
        self.system_ids = system_ids or []
        self.page_ids = page_ids or []
        self.domain_ids = domain_ids or []
        
        # 时间戳
        current_time = datetime.now(timezone.utc).isoformat()
        self.created_at = created_at or current_time
        self.updated_at = updated_at or current_time
        
        # 内部使用字段
        self.content = ""  # 用于搜索和显示的内容描述
        self.metadata = {}  # 额外元数据
        
    def update_timestamp(self):
        """更新时间戳"""
        self.updated_at = datetime.now(timezone.utc).isoformat()
        
    def add_system_id(self, system_id: str):
        """添加系统ID"""
        if system_id not in self.system_ids:
            self.system_ids.append(system_id)
            self.update_timestamp()
            
    def add_page_id(self, page_id: str):
        """添加页面ID"""
        if page_id not in self.page_ids:
            self.page_ids.append(page_id)
            self.update_timestamp()
            
    def add_domain_id(self, domain_id: str):
        """添加领域ID"""
        if domain_id not in self.domain_ids:
            self.domain_ids.append(domain_id)
            self.update_timestamp()
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "knowledge_id": self.knowledge_id,
            "experience_id": self.experience_id,
            "extracted_subjective_knowledge_id": self.extracted_subjective_knowledge_id,
            "operation_steps_id": self.operation_steps_id,
            "system_ids": self.system_ids,
            "page_ids": self.page_ids,
            "domain_ids": self.domain_ids,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            # 内部字段
            "content": self.content,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        """从字典创建节点"""
        node = cls(
            knowledge_id=data.get("knowledge_id"),
            experience_id=data.get("experience_id"),
            extracted_subjective_knowledge_id=data.get("extracted_subjective_knowledge_id"),
            operation_steps_id=data.get("operation_steps_id"),
            system_ids=data.get("system_ids", []),
            page_ids=data.get("page_ids", []),
            domain_ids=data.get("domain_ids", []),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at")
        )
        node.content = data.get("content", "")
        node.metadata = data.get("metadata", {})
        return node


class KnowledgeGraph:
    """知识图谱类"""
    
    def __init__(self):
        """初始化知识图谱"""
        self.knowledge_nodes: Dict[str, KnowledgeNode] = {}  # 以knowledge_id为key的知识节点
        
        # 索引结构
        self.experience_index: Dict[str, Set[str]] = defaultdict(set)  # experience_id -> knowledge_ids
        self.system_index: Dict[str, Set[str]] = defaultdict(set)      # system_id -> knowledge_ids
        self.page_index: Dict[str, Set[str]] = defaultdict(set)        # page_id -> knowledge_ids
        self.domain_index: Dict[str, Set[str]] = defaultdict(set)      # domain_id -> knowledge_ids
        self.content_index: Dict[str, Set[str]] = defaultdict(set)     # keyword -> knowledge_ids
        
    def add_knowledge_node(self, node: KnowledgeNode) -> bool:
        """添加知识节点"""
        try:
            self.knowledge_nodes[node.knowledge_id] = node
            
            # 更新索引
            if node.experience_id:
                self.experience_index[node.experience_id].add(node.knowledge_id)
            
            for system_id in node.system_ids:
                self.system_index[system_id].add(node.knowledge_id)
                
            for page_id in node.page_ids:
                self.page_index[page_id].add(node.knowledge_id)
                
            for domain_id in node.domain_ids:
                self.domain_index[domain_id].add(node.knowledge_id)
            
            # 内容索引
            if node.content:
                keywords = self._extract_keywords(node.content)
                for keyword in keywords:
                    self.content_index[keyword].add(node.knowledge_id)
                
            logger.info(f"添加知识节点: {node.knowledge_id}")
            return True
        except Exception as e:
            logger.error(f"添加知识节点失败: {str(e)}")
            return False
    
    def get_knowledge_by_experience(self, experience_id: str) -> List[KnowledgeNode]:
        """根据经验体ID获取相关知识节点"""
        knowledge_ids = self.experience_index.get(experience_id, set())
        return [self.knowledge_nodes[kid] for kid in knowledge_ids if kid in self.knowledge_nodes]
    
    def get_knowledge_by_system(self, system_id: str) -> List[KnowledgeNode]:
        """根据系统ID获取相关知识节点"""
        knowledge_ids = self.system_index.get(system_id, set())
        return [self.knowledge_nodes[kid] for kid in knowledge_ids if kid in self.knowledge_nodes]
    
    def get_knowledge_by_page(self, page_id: str) -> List[KnowledgeNode]:
        """根据页面ID获取相关知识节点"""
        knowledge_ids = self.page_index.get(page_id, set())
        return [self.knowledge_nodes[kid] for kid in knowledge_ids if kid in self.knowledge_nodes]
    
    def get_knowledge_by_domain(self, domain_id: str) -> List[KnowledgeNode]:
        """根据领域ID获取相关知识节点"""
        knowledge_ids = self.domain_index.get(domain_id, set())
        return [self.knowledge_nodes[kid] for kid in knowledge_ids if kid in self.knowledge_nodes]
    
    def search_knowledge_by_content(self, query: str) -> List[KnowledgeNode]:
        """根据内容搜索知识节点"""
        keywords = self._extract_keywords(query)
        candidate_ids = set()
        
        for keyword in keywords:
            if keyword in self.content_index:
                candidate_ids.update(self.content_index[keyword])
        
        return [self.knowledge_nodes[kid] for kid in candidate_ids if kid in self.knowledge_nodes]
    
    def find_related_knowledge(self, knowledge_id: str) -> Dict[str, List[KnowledgeNode]]:
        """
        查找相关知识节点
        返回与指定知识节点相关的其他知识节点，按关系类型分组
        """
        if knowledge_id not in self.knowledge_nodes:
            return {}
        
        node = self.knowledge_nodes[knowledge_id]
        related = {
            "same_experience": [],
            "same_system": [],
            "same_page": [],
            "same_domain": []
        }
        
        # 同经验体的知识
        if node.experience_id:
            related["same_experience"] = [
                n for n in self.get_knowledge_by_experience(node.experience_id)
                if n.knowledge_id != knowledge_id
            ]
        
        # 同系统的知识
        for system_id in node.system_ids:
            related["same_system"].extend([
                n for n in self.get_knowledge_by_system(system_id)
                if n.knowledge_id != knowledge_id and n not in related["same_system"]
            ])
        
        # 同页面的知识
        for page_id in node.page_ids:
            related["same_page"].extend([
                n for n in self.get_knowledge_by_page(page_id)
                if n.knowledge_id != knowledge_id and n not in related["same_page"]
            ])
        
        # 同领域的知识
        for domain_id in node.domain_ids:
            related["same_domain"].extend([
                n for n in self.get_knowledge_by_domain(domain_id)
                if n.knowledge_id != knowledge_id and n not in related["same_domain"]
            ])
        
        return related
    
    def get_knowledge_chain(self, start_knowledge_id: str, end_knowledge_id: str) -> List[str]:
        """
        获取两个知识节点间的关联链路
        通过共享的系统、页面、领域等建立关联路径
        """
        if (start_knowledge_id not in self.knowledge_nodes or 
            end_knowledge_id not in self.knowledge_nodes):
            return []
        
        start_node = self.knowledge_nodes[start_knowledge_id]
        end_node = self.knowledge_nodes[end_knowledge_id]
        
        # 检查直接关联
        direct_connections = []
        
        # 通过系统关联
        common_systems = set(start_node.system_ids) & set(end_node.system_ids)
        if common_systems:
            direct_connections.append(f"共享系统: {list(common_systems)}")
        
        # 通过页面关联
        common_pages = set(start_node.page_ids) & set(end_node.page_ids)
        if common_pages:
            direct_connections.append(f"共享页面: {list(common_pages)}")
        
        # 通过领域关联
        common_domains = set(start_node.domain_ids) & set(end_node.domain_ids)
        if common_domains:
            direct_connections.append(f"共享领域: {list(common_domains)}")
        
        # 通过经验体关联
        if start_node.experience_id == end_node.experience_id:
            direct_connections.append(f"共享经验体: {start_node.experience_id}")
        
        return direct_connections
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取文本关键词"""
        import re
        words = re.findall(r'[\u4e00-\u9fff\w]+', text.lower())
        return [word for word in words if len(word) > 1]
    
    def get_statistics(self) -> Dict:
        """获取知识图谱统计信息"""
        return {
            "total_knowledge_nodes": len(self.knowledge_nodes),
            "experiences_count": len(self.experience_index),
            "systems_count": len(self.system_index),
            "pages_count": len(self.page_index),
            "domains_count": len(self.domain_index),
            "content_keywords": len(self.content_index)
        }
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "knowledge_nodes": {kid: node.to_dict() for kid, node in self.knowledge_nodes.items()},
            "statistics": self.get_statistics()
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        """从字典创建知识图谱"""
        kg = cls()
        
        # 加载知识节点
        for node_data in data.get("knowledge_nodes", {}).values():
            node = KnowledgeNode.from_dict(node_data)
            kg.add_knowledge_node(node)
        
        return kg


def extract_knowledge_from_experience(experience_data: Dict, experience_id: str = None) -> List[KnowledgeNode]:
    """
    从经验体中提取知识节点
    
    Args:
        experience_data: 经验体数据
        experience_id: 经验体ID
        
    Returns:
        知识节点列表
    """
    knowledge_nodes = []
    exp_id = experience_id or f"exp_{str(uuid.uuid4())[:8]}"
    
    try:
        # 提取系统和页面信息
        systems = set()
        pages = set()
        
        # 从how_behavior_logs中提取页面信息
        if "how_behavior_logs" in experience_data:
            for behavior in experience_data["how_behavior_logs"]:
                if "page" in behavior:
                    pages.add(behavior["page"])
        
        # 从页面信息推断系统（简单规则，可根据实际情况调整）
        for page in pages:
            if "://" in page:  # URL格式
                domain = page.split("://")[1].split("/")[0]
                systems.add(domain)
            else:
                systems.add(f"system_{page}")
        
        # 推断业务领域（基于目标和背景）
        domains = set()
        if "why_structured" in experience_data:
            why_data = experience_data["why_structured"]
            goal = why_data.get("goal", "")
            background = why_data.get("background", "")
            
            # 简单的领域推断逻辑
            combined_text = f"{goal} {background}".lower()
            if any(keyword in combined_text for keyword in ["营销", "广告", "推广"]):
                domains.add("marketing")
            if any(keyword in combined_text for keyword in ["测试", "验证", "质量"]):
                domains.add("testing")
            if any(keyword in combined_text for keyword in ["数据", "分析", "统计"]):
                domains.add("analytics")
            if any(keyword in combined_text for keyword in ["页面", "网站", "前端"]):
                domains.add("frontend")
        
        # 为WHY结构化数据创建知识节点
        if "why_structured" in experience_data:
            why_data = experience_data["why_structured"]
            
            # 目标知识节点
            if "goal" in why_data:
                goal_node = KnowledgeNode(
                    experience_id=exp_id,
                    system_ids=list(systems),
                    page_ids=list(pages),
                    domain_ids=list(domains)
                )
                goal_node.content = f"目标: {why_data['goal']}"
                goal_node.metadata = {"type": "goal", "source": "why_structured"}
                knowledge_nodes.append(goal_node)
            
            # 背景知识节点
            if "background" in why_data:
                bg_node = KnowledgeNode(
                    experience_id=exp_id,
                    system_ids=list(systems),
                    page_ids=list(pages),
                    domain_ids=list(domains)
                )
                bg_node.content = f"背景: {why_data['background']}"
                bg_node.metadata = {"type": "background", "source": "why_structured"}
                knowledge_nodes.append(bg_node)
        
        # 为HOW行为日志创建知识节点
        if "how_behavior_logs" in experience_data:
            for i, behavior in enumerate(experience_data["how_behavior_logs"]):
                behavior_node = KnowledgeNode(
                    experience_id=exp_id,
                    operation_steps_id=f"step_{i+1}",
                    system_ids=list(systems),
                    page_ids=[behavior.get("page", "")] if behavior.get("page") else [],
                    domain_ids=list(domains)
                )
                
                # 构建行为描述
                action = behavior.get("action", "")
                element = behavior.get("element", "")
                intent = behavior.get("intent", "")
                behavior_node.content = f"操作: {action} {element} - {intent}"
                behavior_node.metadata = {
                    "type": "behavior",
                    "source": "how_behavior_logs",
                    "step_index": i,
                    "action": action,
                    "element": element,
                    "intent": intent
                }
                knowledge_nodes.append(behavior_node)
        
        logger.info(f"从经验体 {exp_id} 提取了 {len(knowledge_nodes)} 个知识节点")
        
    except Exception as e:
        logger.error(f"提取知识失败: {str(e)}")
    
    return knowledge_nodes


def build_knowledge_graph_from_experiences(experiences: List[Dict]) -> KnowledgeGraph:
    """
    从经验列表构建知识图谱
    
    Args:
        experiences: 经验体列表
        
    Returns:
        构建的知识图谱
    """
    kg = KnowledgeGraph()
    
    for i, experience in enumerate(experiences):
        exp_id = f"experience_{i+1:03d}"
        knowledge_nodes = extract_knowledge_from_experience(experience, exp_id)
        
        # 添加知识节点
        for node in knowledge_nodes:
            kg.add_knowledge_node(node)
    
    logger.info(f"构建完成，共 {len(kg.knowledge_nodes)} 个知识节点")
    return kg


def save_knowledge_graph(kg: KnowledgeGraph, file_path: str) -> bool:
    """保存知识图谱到文件"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(kg.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"知识图谱已保存到: {file_path}")
        return True
    except Exception as e:
        logger.error(f"保存知识图谱失败: {str(e)}")
        return False


def load_knowledge_graph(file_path: str) -> Optional[KnowledgeGraph]:
    """从文件加载知识图谱"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        kg = KnowledgeGraph.from_dict(data)
        logger.info(f"知识图谱已从 {file_path} 加载")
        return kg
    except Exception as e:
        logger.error(f"加载知识图谱失败: {str(e)}")
        return None
