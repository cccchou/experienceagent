# -*- coding: utf-8 -*-
"""
经验体知识图谱构建模块
专注于多个经验体的联合和关联分析
"""

import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import networkx as nx
from datetime import datetime
import logging

# 导入提示词模块
from prompt import ExperiencePromptEngine

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExperienceEntity:
    """经验体实体"""
    id: str
    name: str
    type: str
    description: str
    source_experience: str
    attributes: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}

@dataclass
class ExperienceRelationship:
    """经验体关系"""
    id: str
    source: str
    target: str
    description: str
    strength: float
    type: str = "RELATED_TO"
    source_experience: str = ""

class ExperienceKnowledgeGraph:
    """经验体知识图谱 - 简化版"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.prompt_engine = ExperiencePromptEngine()
        
        # 图谱存储
        self.entities: Dict[str, ExperienceEntity] = {}
        self.relationships: Dict[str, ExperienceRelationship] = {}
        self.graph = nx.Graph()
        
        # 经验体索引
        self.experience_index: Dict[str, List[str]] = defaultdict(list)
        self.entity_type_index: Dict[str, List[str]] = defaultdict(list)
        
        # 联立结果
        self.unified_entities: Dict[str, List[str]] = {}  # 合并后的实体组
        self.cross_relationships: List[ExperienceRelationship] = []  # 跨经验体关系
        
        logger.info("经验体知识图谱初始化完成")
    
    async def extract_experience_graph(self, experience_data: Dict[str, Any], 
                                     experience_id: str) -> Dict[str, Any]:
        """
        从单个经验体中提取实体和关系
        """
        logger.info(f"开始提取经验体 {experience_id} 的图谱结构")
        
        # 1. 调用提示词引擎生成提取提示词
        extraction_prompt = self.prompt_engine.get_entity_extraction_prompt(
            experience_data, experience_id
        )
        
        # 2. 调用LLM进行实体关系提取
        if not self.llm_client:
            logger.warning("未配置LLM客户端，返回提示词供外部调用")
            return {
                "status": "pending",
                "prompt": extraction_prompt,
                "experience_id": experience_id
            }
        
        try:
            llm_response = await self._call_llm(extraction_prompt)
            entities, relationships = self.prompt_engine.parse_extraction_result(llm_response)
            
            # 3. 存储到图谱中
            extracted_entities = []
            extracted_relationships = []
            
            for entity_data in entities:
                entity = self._create_entity(entity_data, experience_id)
                self.entities[entity.id] = entity
                extracted_entities.append(entity)
                
                # 更新索引
                self.experience_index[experience_id].append(entity.id)
                self.entity_type_index[entity.type].append(entity.id)
                
                # 添加到NetworkX图
                self.graph.add_node(entity.id, **{
                    "name": entity.name,
                    "type": entity.type,
                    "description": entity.description,
                    "experience": experience_id
                })
            
            for rel_data in relationships:
                relationship = self._create_relationship(rel_data, experience_id)
                self.relationships[relationship.id] = relationship
                extracted_relationships.append(relationship)
                
                # 添加边到NetworkX图
                if relationship.source in self.entities and relationship.target in self.entities:
                    self.graph.add_edge(
                        relationship.source, relationship.target,
                        weight=relationship.strength,
                        description=relationship.description,
                        type=relationship.type,
                        id=relationship.id
                    )
            
            logger.info(f"成功提取 {len(extracted_entities)} 个实体，{len(extracted_relationships)} 个关系")
            
            return {
                "status": "success",
                "experience_id": experience_id,
                "entities": extracted_entities,
                "relationships": extracted_relationships,
                "entity_count": len(extracted_entities),
                "relationship_count": len(extracted_relationships)
            }
            
        except Exception as e:
            logger.error(f"提取经验体图谱时出错: {e}")
            return {
                "status": "error",
                "experience_id": experience_id,
                "error": str(e)
            }
    
    async def unify_experiences(self, experience_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        联立多个经验体 - 核心功能
        
        Args:
            experience_results: 多个经验体的提取结果
            
        Returns:
            联立结果
        """
        logger.info("开始联立多个经验体")
        
        # 1. 识别相同概念的实体
        unified_entities = self._identify_unified_entities()
        
        # 2. 发现跨经验体关系
        cross_relationships = await self._discover_cross_relationships()
        
        # 3. 生成联立分析报告
        unification_report = await self._generate_unification_report(
            unified_entities, cross_relationships
        )
        
        # 4. 存储联立结果
        self.unified_entities = unified_entities
        self.cross_relationships = cross_relationships
        
        return {
            "status": "success",
            "unified_entities": unified_entities,
            "cross_relationships": cross_relationships,
            "unification_report": unification_report,
            "total_experiences": len(self.experience_index),
            "total_entities": len(self.entities),
            "total_relationships": len(self.relationships)
        }
    
    def find_similar_experiences(self, target_experience_id: str, 
                               similarity_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        寻找与目标经验体相似的其他经验体（同步版本）
        
        Args:
            target_experience_id: 目标经验体ID
            similarity_threshold: 相似度阈值
            
        Returns:
            相似经验体列表
        """
        logger.info(f"寻找与经验体 {target_experience_id} 相似的经验体")
        
        target_entities = self.experience_index.get(target_experience_id, [])
        if not target_entities:
            return []
        
        similar_experiences = []
        
        for exp_id, entities in self.experience_index.items():
            if exp_id == target_experience_id:
                continue
            
            # 计算相似度
            similarity_metrics = self._calculate_experience_similarity(
                target_entities, entities
            )
            
            if similarity_metrics["overall_similarity"] >= similarity_threshold:
                similar_experiences.append({
                    "experience_id": exp_id,
                    "similarity_metrics": similarity_metrics,
                    "can_unify": similarity_metrics["overall_similarity"] > 0.5
                })
        
        # 按相似度排序
        similar_experiences.sort(
            key=lambda x: x["similarity_metrics"]["overall_similarity"], 
            reverse=True
        )
        
        return similar_experiences
    
    def get_unified_knowledge_base(self) -> Dict[str, Any]:
        """
        获取联立后的知识库
        
        Returns:
            统一的知识库结构
        """
        knowledge_base = {
            "unified_concepts": {},
            "shared_patterns": [],
            "cross_experience_flows": [],
            "consolidated_rules": []
        }
        
        # 1. 统一概念
        for concept_name, entity_ids in self.unified_entities.items():
            entities = [self.entities[eid] for eid in entity_ids if eid in self.entities]
            if entities:
                knowledge_base["unified_concepts"][concept_name] = {
                    "description": self._merge_entity_descriptions(entities),
                    "type": entities[0].type,
                    "source_experiences": [e.source_experience for e in entities],
                    "occurrence_count": len(entities)
                }
        
        # 2. 共享模式
        knowledge_base["shared_patterns"] = self._identify_shared_patterns()
        
        # 3. 跨经验体流程
        knowledge_base["cross_experience_flows"] = self._identify_cross_flows()
        
        # 4. 合并规则
        knowledge_base["consolidated_rules"] = self._consolidate_rules()
        
        return knowledge_base
    
    def export_unified_graph(self) -> Dict[str, Any]:
        """
        导出联立后的图谱数据
        
        Returns:
            完整的联立图谱数据
        """
        return {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "created_by": "cccchou",
                "total_experiences": len(self.experience_index),
                "unified_entity_groups": len(self.unified_entities),
                "cross_relationships": len(self.cross_relationships)
            },
            "experiences": list(self.experience_index.keys()),
            "unified_entities": self.unified_entities,
            "cross_relationships": [self._relationship_to_dict(r) for r in self.cross_relationships],
            "knowledge_base": self.get_unified_knowledge_base(),
            "entity_distribution": self._get_entity_distribution(),
            "relationship_distribution": self._get_relationship_distribution()
        }
    
    # 私有方法
    
    async def _call_llm(self, prompt: str) -> str:
        """调用LLM"""
        if hasattr(self.llm_client, 'chat'):
            if asyncio.iscoroutinefunction(self.llm_client.chat):
                return await self.llm_client.chat(prompt)
            else:
                return self.llm_client.chat(prompt)
        elif hasattr(self.llm_client, 'generate'):
            if asyncio.iscoroutinefunction(self.llm_client.generate):
                return await self.llm_client.generate(prompt)
            else:
                return self.llm_client.generate(prompt)
        else:
            raise ValueError("LLM客户端必须实现chat或generate方法")
    
    def _create_entity(self, entity_data: Dict[str, Any], experience_id: str) -> ExperienceEntity:
        """创建实体对象"""
        entity_id = f"{experience_id}_{entity_data['name'].replace(' ', '_').replace('/', '_')}"
        return ExperienceEntity(
            id=entity_id,
            name=entity_data['name'],
            type=entity_data['type'],
            description=entity_data['description'],
            source_experience=experience_id,
            attributes=entity_data.get('attributes', {})
        )
    
    def _create_relationship(self, rel_data: Dict[str, Any], experience_id: str) -> ExperienceRelationship:
        """创建关系对象"""
        source_id = f"{experience_id}_{rel_data['source'].replace(' ', '_').replace('/', '_')}"
        target_id = f"{experience_id}_{rel_data['target'].replace(' ', '_').replace('/', '_')}"
        rel_id = f"{source_id}-{target_id}"
        
        return ExperienceRelationship(
            id=rel_id,
            source=source_id,
            target=target_id,
            description=rel_data['description'],
            strength=rel_data.get('strength', 1.0),
            type=rel_data.get('type', 'RELATED_TO'),
            source_experience=experience_id
        )
    
    def _identify_unified_entities(self) -> Dict[str, List[str]]:
        """识别可以统一的实体"""
        unified_groups = defaultdict(list)
        
        # 按名称和类型进行分组
        entity_groups = defaultdict(list)
        
        for entity in self.entities.values():
            # 标准化实体名称
            normalized_name = entity.name.lower().strip()
            key = f"{entity.type}_{normalized_name}"
            entity_groups[key].append(entity.id)
        
        # 找出跨经验体的相同实体
        for key, entity_ids in entity_groups.items():
            if len(entity_ids) > 1:
                # 检查是否来自不同经验体
                experiences = set()
                for eid in entity_ids:
                    entity = self.entities[eid]
                    experiences.add(entity.source_experience)
                
                if len(experiences) > 1:  # 来自不同经验体
                    concept_name = key.split('_', 1)[1]  # 移除类型前缀
                    unified_groups[concept_name] = entity_ids
        
        return dict(unified_groups)
    
    async def _discover_cross_relationships(self) -> List[ExperienceRelationship]:
        """发现跨经验体关系"""
        cross_rels = []
        
        if not self.llm_client:
            logger.warning("未配置LLM客户端，跳过跨关系发现")
            return cross_rels
        
        # 使用提示词引擎分析跨经验体实体关系
        cross_analysis_prompt = self.prompt_engine.get_cross_relationship_prompt(
            list(self.entities.values())
        )
        
        try:
            analysis_result = await self._call_llm(cross_analysis_prompt)
            # 解析跨关系结果
            _, cross_relationships = self.prompt_engine.parse_extraction_result(analysis_result)
            
            for rel_data in cross_relationships:
                # 创建跨经验体关系
                cross_rel = ExperienceRelationship(
                    id=f"cross_{len(cross_rels)}",
                    source=rel_data['source'],
                    target=rel_data['target'],
                    description=rel_data['description'],
                    strength=rel_data.get('strength', 1.0),
                    type=rel_data.get('type', 'CROSS_EXPERIENCE'),
                    source_experience="CROSS"
                )
                cross_rels.append(cross_rel)
                
        except Exception as e:
            logger.error(f"跨关系分析时出错: {e}")
        
        return cross_rels
    
    async def _generate_unification_report(self, unified_entities: Dict[str, List[str]], 
                                         cross_relationships: List[ExperienceRelationship]) -> str:
        """生成联立分析报告"""
        if not self.llm_client:
            return "未配置LLM客户端，无法生成详细报告"
        
        # 使用提示词引擎生成报告提示词
        report_prompt = self.prompt_engine.get_unification_report_prompt(
            unified_entities, cross_relationships, list(self.experience_index.keys())
        )
        
        try:
            report = await self._call_llm(report_prompt)
            return report
        except Exception as e:
            logger.error(f"生成联立报告时出错: {e}")
            return f"报告生成失败: {str(e)}"
    
    def _calculate_experience_similarity(self, entities1: List[str], 
                                       entities2: List[str]) -> Dict[str, float]:
        """计算经验体相似度"""
        if not entities1 or not entities2:
            return {"overall_similarity": 0.0, "entity_overlap": 0.0, "type_similarity": 0.0}
        
        # 1. 实体名称重叠度
        names1 = set()
        names2 = set()
        types1 = set()
        types2 = set()
        
        for eid in entities1:
            entity = self.entities.get(eid)
            if entity:
                names1.add(entity.name.lower().strip())
                types1.add(entity.type)
        
        for eid in entities2:
            entity = self.entities.get(eid)
            if entity:
                names2.add(entity.name.lower().strip())
                types2.add(entity.type)
        
        # 计算重叠度
        name_overlap = len(names1 & names2) / len(names1 | names2) if (names1 | names2) else 0
        type_overlap = len(types1 & types2) / len(types1 | types2) if (types1 | types2) else 0
        
        # 综合相似度
        overall_similarity = (name_overlap * 0.7 + type_overlap * 0.3)
        
        return {
            "overall_similarity": overall_similarity,
            "entity_overlap": name_overlap,
            "type_similarity": type_overlap
        }
    
    def _merge_entity_descriptions(self, entities: List[ExperienceEntity]) -> str:
        """合并实体描述"""
        descriptions = [e.description for e in entities if e.description]
        if not descriptions:
            return "无描述"
        
        if len(descriptions) == 1:
            return descriptions[0]
        
        # 简单合并，去重
        unique_parts = []
        for desc in descriptions:
            if desc not in unique_parts:
                unique_parts.append(desc)
        
        return " | ".join(unique_parts)
    
    def _identify_shared_patterns(self) -> List[Dict[str, Any]]:
        """识别共享模式"""
        patterns = []
        
        # 找出在多个经验体中都出现的实体类型组合
        type_combinations = defaultdict(set)
        
        for exp_id, entity_ids in self.experience_index.items():
            entity_types = set()
            for eid in entity_ids:
                entity = self.entities.get(eid)
                if entity:
                    entity_types.add(entity.type)
            
            # 生成类型组合
            type_list = sorted(list(entity_types))
            for i in range(len(type_list)):
                for j in range(i+1, len(type_list)):
                    combo = f"{type_list[i]}-{type_list[j]}"
                    type_combinations[combo].add(exp_id)
        
        # 找出跨经验体的模式
        for combo, experiences in type_combinations.items():
            if len(experiences) > 1:
                patterns.append({
                    "pattern": combo,
                    "experiences": list(experiences),
                    "frequency": len(experiences)
                })
        
        return sorted(patterns, key=lambda x: x["frequency"], reverse=True)
    
    def _identify_cross_flows(self) -> List[Dict[str, Any]]:
        """识别跨经验体流程"""
        flows = []
        
        for rel in self.cross_relationships:
            # 找出源和目标实体所属的经验体
            source_entity = None
            target_entity = None
            
            for entity in self.entities.values():
                if entity.name == rel.source:
                    source_entity = entity
                elif entity.name == rel.target:
                    target_entity = entity
            
            if source_entity and target_entity:
                flows.append({
                    "from_experience": source_entity.source_experience,
                    "to_experience": target_entity.source_experience,
                    "relationship": rel.description,
                    "strength": rel.strength
                })
        
        return flows
    
    def _consolidate_rules(self) -> List[str]:
        """合并规则"""
        rules = []
        
        # 从所有RULE类型的实体中提取规则
        for entity in self.entities.values():
            if entity.type == "RULE":
                rules.append(f"[{entity.source_experience}] {entity.description}")
        
        return rules
    
    def _get_entity_distribution(self) -> Dict[str, int]:
        """获取实体类型分布"""
        distribution = {}
        for entity in self.entities.values():
            entity_type = entity.type
            distribution[entity_type] = distribution.get(entity_type, 0) + 1
        return distribution
    
    def _get_relationship_distribution(self) -> Dict[str, int]:
        """获取关系类型分布"""
        distribution = {}
        for rel in self.relationships.values():
            rel_type = rel.type
            distribution[rel_type] = distribution.get(rel_type, 0) + 1
        return distribution
    
    def _relationship_to_dict(self, relationship: ExperienceRelationship) -> Dict[str, Any]:
        """关系转字典"""
        return {
            "id": relationship.id,
            "source": relationship.source,
            "target": relationship.target,
            "description": relationship.description,
            "strength": relationship.strength,
            "type": relationship.type,
            "source_experience": relationship.source_experience
        }

# 便捷函数

async def unify_experience_knowledge(experiences: List[Dict[str, Any]], 
                                   llm_client=None) -> ExperienceKnowledgeGraph:
    """
    联立多个经验体的便捷函数
    
    Args:
        experiences: 经验体数据列表，每个包含id和data字段
        llm_client: LLM客户端
        
    Returns:
        联立后的知识图谱
    """
    
    kg = ExperienceKnowledgeGraph(llm_client)
    
    # 1. 提取各个经验体的图谱
    extraction_results = []
    for i, exp in enumerate(experiences):
        exp_id = exp.get('id', f'experience_{i}')
        exp_data = exp.get('data', exp)
        
        result = await kg.extract_experience_graph(exp_data, exp_id)
        extraction_results.append(result)
    
    # 2. 联立经验体
    unification_result = await kg.unify_experiences(extraction_results)
    
    logger.info("经验体联立完成")
    return kg

# 使用示例
if __name__ == "__main__":
    async def main():
        # 示例经验体数据
        sample_experiences = [
            {
                "id": "marketing_validation_A",
                "data": {
                    "why_structured": {
                        "goal": "建立营销页面验证框架",
                        "constraints": ["基于HTML结构", "多终端兼容"]
                    },
                    "how_behavior_logs": [
                        {
                            "page": "验证页",
                            "action": "点击",
                            "element": "验证按钮",
                            "intent": "执行验证"
                        }
                    ]
                }
            },
            {
                "id": "marketing_validation_B", 
                "data": {
                    "why_structured": {
                        "goal": "页面元素自动检测",
                        "constraints": ["基于HTML结构", "实时监控"]
                    },
                    "how_behavior_logs": [
                        {
                            "page": "监控页",
                            "action": "扫描",
                            "element": "页面元素",
                            "intent": "自动检测"
                        }
                    ]
                }
            }
        ]
        
        # 联立经验体
        kg = await unify_experience_knowledge(sample_experiences)
        
        # 导出联立结果
        unified_data = kg.export_unified_graph()
        print(f"联立完成：{json.dumps(unified_data['metadata'], ensure_ascii=False, indent=2)}")
        
        # 查看统一的知识库
        knowledge_base = kg.get_unified_knowledge_base()
        print(f"统一知识库概念数：{len(knowledge_base['unified_concepts'])}")
    
    # 运行示例
    asyncio.run(main())