# -*- coding: utf-8 -*-
"""
经验体提示词工程模块 - 简化版
专门负责生成经验体联立相关的提示词
"""

import json
from typing import List, Dict, Any, Optional

# GraphRAG风格的分隔符
TUPLE_DELIMITER = "<|>"
RECORD_DELIMITER = "##"
COMPLETION_DELIMITER = "<|COMPLETE|>"

class ExperiencePromptEngine:
    """经验体提示词引擎"""
    
    def __init__(self):
        # 经验体特定的实体类型
        self.entity_types = [
            "GOAL",           # 目标实体
            "CONSTRAINT",     # 约束条件
            "ACTION",         # 行为动作
            "ELEMENT",        # 页面元素
            "RULE",           # 检查规则
            "OUTCOME",        # 预期结果
            "INTENT",         # 用户意图
            "PROCESS",        # 流程步骤
            "REQUIREMENT",    # 需求要求
            "SCENARIO",       # 场景
            "CONDITION",      # 条件
            "RESOURCE",       # 资源
            "METHOD"          # 方法
        ]
    
    def get_entity_extraction_prompt(self, experience_data: Dict[str, Any], 
                                   experience_id: str) -> str:
        """
        生成实体关系提取提示词
        """
        entity_types_str = ', '.join(self.entity_types)
        
        prompt = f"""
-目标-
从经验体数据中提取实体和关系，用于后续的经验体联立分析。

-步骤-
1. 识别所有实体，包括：
- entity_name: 实体名称，使用统一规范的命名
- entity_type: 类型，从以下选择：[{entity_types_str}]
- entity_description: 详细描述

格式：("entity"{TUPLE_DELIMITER}<entity_name>{TUPLE_DELIMITER}<entity_type>{TUPLE_DELIMITER}<entity_description>)

2. 识别实体间关系：
- source_entity: 源实体名称
- target_entity: 目标实体名称  
- relationship_description: 关系描述
- relationship_strength: 关系强度(1-10)

格式：("relationship"{TUPLE_DELIMITER}<source_entity>{TUPLE_DELIMITER}<target_entity>{TUPLE_DELIMITER}<relationship_description>{TUPLE_DELIMITER}<relationship_strength>)

3. 使用 **{RECORD_DELIMITER}** 分隔，最后输出 {COMPLETION_DELIMITER}

######################
-示例-
######################
经验体ID: validation_demo
数据: {{"goal": "页面验证", "action": "点击按钮"}}
输出:
("entity"{TUPLE_DELIMITER}页面验证{TUPLE_DELIMITER}GOAL{TUPLE_DELIMITER}验证页面元素和功能的目标){RECORD_DELIMITER}
("entity"{TUPLE_DELIMITER}点击按钮{TUPLE_DELIMITER}ACTION{TUPLE_DELIMITER}用户在页面上执行点击操作){RECORD_DELIMITER}
("relationship"{TUPLE_DELIMITER}页面验证{TUPLE_DELIMITER}点击按钮{TUPLE_DELIMITER}验证目标通过点击操作实现{TUPLE_DELIMITER}8){RECORD_DELIMITER}
{COMPLETION_DELIMITER}

######################
-实际数据-
######################
经验体ID: {experience_id}
经验体数据:
{json.dumps(experience_data, ensure_ascii=False, indent=2)}
######################
输出:"""
        
        return prompt
    
    def get_cross_relationship_prompt(self, entities: List[Any]) -> str:
        """
        生成跨经验体关系发现提示词
        """
        # 按经验体分组实体
        experience_groups = {}
        for entity in entities:
            exp_id = getattr(entity, 'source_experience', 'unknown')
            if exp_id not in experience_groups:
                experience_groups[exp_id] = []
            experience_groups[exp_id].append({
                'name': getattr(entity, 'name', 'unknown'),
                'type': getattr(entity, 'type', 'unknown'),
                'description': getattr(entity, 'description', '')[:100]
            })
        
        prompt = f"""
-目标-
分析不同经验体中的实体，找出可以建立联系的跨经验体关系。

-分析重点-
1. 概念相似：不同经验体中名称或功能相似的实体
2. 流程衔接：一个经验体的输出连接另一个经验体的输入
3. 资源共享：不同经验体使用相同的资源或工具
4. 目标支撑：不同经验体为相同目标提供支持

-输出格式-
对于每个跨经验体关系：
("cross_relationship"{TUPLE_DELIMITER}<实体A名称>{TUPLE_DELIMITER}<实体B名称>{TUPLE_DELIMITER}<关系描述>{TUPLE_DELIMITER}<关系强度>{TUPLE_DELIMITER}<关系类型>)

关系类型：SUPPORTS(支撑)、SHARES(共享)、CONNECTS(连接)、COMPLEMENTS(互补)

使用 **{RECORD_DELIMITER}** 分隔。

######################
-经验体数据-
######################
{json.dumps(experience_groups, ensure_ascii=False, indent=2)}
######################
输出:"""
        
        return prompt
    
    def get_unification_report_prompt(self, unified_entities: Dict[str, List[str]], 
                                    cross_relationships: List[Any],
                                    experience_ids: List[str]) -> str:
        """
        生成联立分析报告提示词
        """
        prompt = f"""
-目标-
为多个经验体的联立结果生成分析报告，突出联立的价值和应用建议。

-报告内容-
1. 联立概况：总结参与联立的经验体数量和主要特征
2. 统一概念：分析哪些概念在多个经验体中出现，说明其重要性
3. 关联分析：描述经验体间的关系和相互影响
4. 价值评估：评估联立后的整体价值和优势
5. 应用建议：提出如何应用联立后的知识体系

-输出格式-
请生成结构化的中文报告，包含上述5个部分。

######################
-联立数据-
######################
参与经验体: {', '.join(experience_ids)}
统一概念数量: {len(unified_entities)}
跨关系数量: {len(cross_relationships)}

统一概念详情:
{json.dumps(unified_entities, ensure_ascii=False, indent=2)}
######################
输出报告:"""
        
        return prompt
    
    def parse_extraction_result(self, llm_output: str) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        解析LLM的实体关系提取结果
        """
        entities = []
        relationships = []
        
        try:
            # 清理输出
            content = llm_output.replace(COMPLETION_DELIMITER, "").strip()
            
            # 按记录分隔符分割
            records = content.split(RECORD_DELIMITER)
            
            for record in records:
                record = record.strip()
                if not record:
                    continue
                
                # 解析实体
                if record.startswith('("entity"'):
                    entity_data = self._parse_entity_record(record)
                    if entity_data:
                        entities.append(entity_data)
                
                # 解析关系
                elif record.startswith('("relationship"') or record.startswith('("cross_relationship"'):
                    relationship_data = self._parse_relationship_record(record)
                    if relationship_data:
                        relationships.append(relationship_data)
        
        except Exception as e:
            print(f"解析提取结果时出错: {e}")
        
        return entities, relationships
    
    def _parse_entity_record(self, record: str) -> Optional[Dict[str, Any]]:
        """解析实体记录"""
        try:
            start = record.find('(')
            end = record.rfind(')')
            if start == -1 or end == -1:
                return None
            
            inner = record[start+1:end]
            parts = inner.split(TUPLE_DELIMITER)
            
            if len(parts) >= 4:
                return {
                    'name': parts[1].strip().strip('"'),
                    'type': parts[2].strip().strip('"'),
                    'description': parts[3].strip().strip('"'),
                    'attributes': {}
                }
        
        except Exception as e:
            print(f"解析实体记录出错: {e}")
        
        return None
    
    def _parse_relationship_record(self, record: str) -> Optional[Dict[str, Any]]:
        """解析关系记录"""
        try:
            start = record.find('(')
            end = record.rfind(')')
            if start == -1 or end == -1:
                return None
            
            inner = record[start+1:end]
            parts = inner.split(TUPLE_DELIMITER)
            
            if len(parts) >= 5:
                strength = 1.0
                try:
                    strength = float(parts[4].strip().strip('"'))
                except ValueError:
                    strength = 1.0
                
                rel_type = "RELATED_TO"
                if len(parts) > 5:
                    rel_type = parts[5].strip().strip('"')
                elif record.startswith('("cross_relationship"'):
                    rel_type = "CROSS_EXPERIENCE"
                
                return {
                    'source': parts[1].strip().strip('"'),
                    'target': parts[2].strip().strip('"'),
                    'description': parts[3].strip().strip('"'),
                    'strength': strength,
                    'type': rel_type
                }
        
        except Exception as e:
            print(f"解析关系记录出错: {e}")
        
        return None

# 便捷函数
def create_extraction_prompt(experience_data: Dict[str, Any], experience_id: str) -> str:
    """创建实体关系提取提示词的便捷函数"""
    engine = ExperiencePromptEngine()
    return engine.get_entity_extraction_prompt(experience_data, experience_id)

def parse_llm_response(llm_output: str) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """解析LLM响应的便捷函数"""
    engine = ExperiencePromptEngine()
    return engine.parse_extraction_result(llm_output)

# 测试代码
if __name__ == "__main__":
    engine = ExperiencePromptEngine()
    
    sample_data = {
        "why_structured": {
            "goal": "页面验证测试",
            "constraints": ["HTML结构", "多终端"]
        },
        "how_behavior_logs": [
            {
                "page": "测试页",
                "action": "点击", 
                "element": "验证按钮",
                "intent": "启动验证"
            }
        ]
    }
    
    prompt = engine.get_entity_extraction_prompt(sample_data, "test_experience")
    print("生成的提取提示词:")
    print(prompt[:500] + "...")