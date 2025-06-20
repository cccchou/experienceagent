"""
经验片段评分器
基于知识图谱和语义相关性进行评估，支持交互式反馈
"""

from typing import Dict, List, Any, Optional, Tuple
from openai import OpenAI
import json
import logging
from experienceagent.knowledge_graph import KnowledgePoint, ExperienceGraph

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ExperienceEvaluator")

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


class ExperienceEvaluator:
    """
    经验片段评价器 - 基于知识图谱进行语义评估
    实现交互式评估流程，通过语义理解提升评估质量
    """
    def __init__(self, knowledge_graph: Optional[ExperienceGraph] = None):
        """
        初始化评估器
        
        Args:
            knowledge_graph: 知识图谱实例，用于语义相关性分析
        """
        self.knowledge_graph = knowledge_graph
        
        # 分类器权重配置
        self.evaluation_criteria = {
            "WHY": {
                "goal_clarity": 0.3,        # 目标清晰度
                "relevance": 0.3,           # 与任务相关性
                "contextual_fit": 0.2,      # 上下文适配度
                "feasibility": 0.2          # 可行性
            },
            "HOW": {
                "completeness": 0.25,       # 步骤完整度
                "precision": 0.25,          # 操作精确度
                "logical_flow": 0.2,        # 逻辑流程性
                "edge_case_handling": 0.15, # 边缘情况处理
                "efficiency": 0.15          # 效率
            },
            "CHECK": {
                "coverage": 0.3,            # 规则覆盖面
                "specificity": 0.25,        # 具体性
                "testability": 0.25,        # 可测试性
                "error_handling": 0.2       # 错误处理
            }
        }
        
        # 上次评估的交互式上下文
        self.interactive_context = {}
        
    def evaluate_fragment(self, fragment, interactive: bool = False) -> Dict[str, Any]:
        """
        评估单个经验片段
        
        Args:
            fragment: 经验片段对象
            interactive: 是否启用交互式评估
            
        Returns:
            评估结果字典
        """
        if not hasattr(fragment, 'frag_type') or not hasattr(fragment, 'data'):
            return {"score": 0.1, "feedback": ["无效的片段格式"]}
        
        frag_type = fragment.frag_type
        frag_data = fragment.data
        
        # 首先在知识图谱中找到相关节点
        related_nodes = self._find_related_knowledge_nodes(frag_type, frag_data)
        
        # 根据片段类型选择评估方法
        if frag_type == "WHY":
            result = self._evaluate_why_fragment(frag_data, related_nodes, interactive)
        elif frag_type == "HOW":
            result = self._evaluate_how_fragment(frag_data, related_nodes, interactive)
        elif frag_type == "CHECK":
            result = self._evaluate_check_fragment(frag_data, related_nodes, interactive)
        else:
            # 通用评估方法
            result = self._evaluate_generic_fragment(frag_data, related_nodes, interactive)
        
        # 如果知识图谱存在且有相关节点，增加相关性分析
        if related_nodes:
            result["related_knowledge"] = [
                {"id": node.id, "type": node.type, "relevance": score}
                for node, score in related_nodes[:3]  # 只返回前3个相关节点
            ]
        
        return result
    
    def _find_related_knowledge_nodes(self, frag_type: str, frag_data: Dict) -> List[Tuple[Any, float]]:
        """
        在知识图谱中查找与片段相关的节点
        
        Args:
            frag_type: 片段类型
            frag_data: 片段数据
            
        Returns:
            [(知识点节点, 相关性分数)] 的列表，按相关性降序排序
        """
        if not self.knowledge_graph:
            return []
            
        # 将片段数据转换为文本，用于相关性匹配
        frag_text = ""
        
        if frag_type == "WHY":
            if "goal" in frag_data:
                frag_text += f"目标: {frag_data['goal']} "
            if "background" in frag_data:
                frag_text += f"背景: {frag_data['background']} "
            if "constraints" in frag_data and isinstance(frag_data["constraints"], list):
                frag_text += f"约束: {', '.join(frag_data['constraints'])} "
            if "expected_outcome" in frag_data:
                frag_text += f"预期效果: {frag_data['expected_outcome']}"
                
        elif frag_type == "HOW":
            steps = frag_data.get("steps", [])
            for i, step in enumerate(steps):
                if isinstance(step, dict) and "description" in step:
                    frag_text += f"步骤{i+1}: {step['description']} "
                else:
                    frag_text += f"步骤{i+1}: {str(step)} "
                    
        elif frag_type == "CHECK":
            rules = frag_data.get("rules", [])
            for i, rule in enumerate(rules):
                frag_text += f"规则{i+1}: {str(rule)} "
                
        else:
            # 通用转换
            frag_text = json.dumps(frag_data, ensure_ascii=False)
        
        # 尝试使用知识图谱的方法找相关节点
        related_nodes = []
        
        try:
            # 如果知识图谱有查询相关节点的方法，使用它
            if hasattr(self.knowledge_graph, "query_related_nodes"):
                related_nodes = self.knowledge_graph.query_related_nodes(frag_text)
            
            # 没有现成方法，使用语义比较（简化实现）
            else:
                all_nodes = self.knowledge_graph.get_all_nodes() if hasattr(self.knowledge_graph, "get_all_nodes") else []
                
                # 用LLM计算相关性 (如果节点数量多，这种方式效率较低)
                if len(all_nodes) > 0:
                    nodes_info = []
                    for i, node in enumerate(all_nodes):
                        node_info = {
                            "id": i,
                            "type": node.type,
                            "content": node.content
                        }
                        nodes_info.append(node_info)
                    
                    # 使用LLM计算相关性
                    prompt = f"""
请分析下面的片段内容与知识点之间的相关性，返回最相关的知识点ID和相关性分数(0-1之间)。

片段内容: {frag_text}

知识点列表:
{json.dumps(nodes_info, ensure_ascii=False)}

返回格式(JSON):
[
  {{"id": 知识点ID, "relevance": 相关性分数}},
  {{"id": 知识点ID, "relevance": 相关性分数}},
  ...
]
只返回相关性大于0.5的结果，按相关性降序排序。
"""
                    try:
                        response = call_openai(prompt)
                        matches = json.loads(response)
                        
                        # 构建结果
                        for match in matches:
                            node_id = match.get("id")
                            if isinstance(node_id, int) and 0 <= node_id < len(all_nodes):
                                related_nodes.append((all_nodes[node_id], match.get("relevance", 0)))
                    except Exception as e:
                        logger.error(f"计算语义相关性失败: {str(e)}")
                
        except Exception as e:
            logger.error(f"查询相关知识节点失败: {str(e)}")
                
        # 按相关性排序
        related_nodes.sort(key=lambda x: x[1], reverse=True)
        return related_nodes
    
    def _evaluate_why_fragment(self, data: Dict, related_nodes: List, interactive: bool) -> Dict:
        """评估WHY类型片段"""
        if not data:
            return {"score": 0.1, "feedback": ["WHY片段数据为空"]}
        
        criteria = self.evaluation_criteria["WHY"]
        scores = {}
        feedback = []
        suggestions = []
        
        # 从知识图谱中提取相关背景知识
        context_knowledge = ""
        if related_nodes:
            for node, score in related_nodes[:3]:  # 取前3个最相关的节点
                if node.type in ["GOAL", "BACKGROUND", "CONSTRAINT", "DOMAIN"]:
                    context_knowledge += f"- {node.content} (相关度: {score:.2f})\n"
        
        # 使用LLM进行综合评估
        prompt = f"""
请作为经验评估专家，评估以下"WHY"类型的经验片段质量。

片段内容:
- 目标: {data.get('goal', '未提供')}
- 背景: {data.get('background', '未提供')}
- 约束条件: {', '.join(data.get('constraints', ['未提供']))}
- 预期效果: {data.get('expected_outcome', '未提供')}

相关知识背景:
{context_knowledge if context_knowledge else "无相关背景知识"}

请从以下维度进行评估(1-10分):
1. 目标清晰度: 目标描述是否清晰明确
2. 相关性: 与任务领域的相关程度
3. 上下文适配度: 与已有知识背景的一致性
4. 可行性: 目标在约束条件下的可实现程度

请以JSON格式返回评估结果:
{{
  "scores": {{
    "goal_clarity": 评分,
    "relevance": 评分,
    "contextual_fit": 评分,
    "feasibility": 评分
  }},
  "overall_score": 0-1之间的总体评分,
  "feedback": ["反馈1", "反馈2", ...],
  "suggestions": ["建议1", "建议2", ...]
}}
"""
        
        try:
            response = call_openai(prompt)
            result = json.loads(response)
            
            if "scores" in result:
                scores = result["scores"]
            
            if "feedback" in result and isinstance(result["feedback"], list):
                feedback = result["feedback"]
                
            if "suggestions" in result and isinstance(result["suggestions"], list):
                suggestions = result["suggestions"]
                
            # 如果LLM返回了整体评分，使用它
            if "overall_score" in result:
                overall_score = float(result["overall_score"])
                # 确保分数在0-1范围内
                overall_score = max(0.0, min(1.0, overall_score))
            else:
                # 否则计算加权平均分
                overall_score = sum(
                    scores.get(criterion, 5) / 10 * weight 
                    for criterion, weight in criteria.items()
                )
            
            return {
                "score": overall_score,
                "detail_scores": scores,
                "feedback": feedback,
                "suggestions": suggestions
            }
            
        except Exception as e:
            logger.error(f"WHY片段评估失败: {str(e)}")
            # 备用评估逻辑
            completeness = self._calculate_completeness(data)
            return {
                "score": completeness * 0.6 + 0.2,  # 基础分0.2，最高0.8
                "feedback": ["自动评估: 内容完整度 {:.0f}%".format(completeness * 100)],
                "suggestions": ["请完善WHY片段内容"]
            }
    
    def _evaluate_how_fragment(self, data: Dict, related_nodes: List, interactive: bool) -> Dict:
        """评估HOW类型片段"""
        if not data:
            return {"score": 0.1, "feedback": ["HOW片段数据为空"]}
            
        criteria = self.evaluation_criteria["HOW"]
        scores = {}
        feedback = []
        suggestions = []
        
        # 从知识图谱中提取相关操作知识
        context_knowledge = ""
        if related_nodes:
            for node, score in related_nodes[:3]:  # 取前3个最相关的节点
                if node.type in ["OPERATION", "PROCEDURE", "TOOL", "INTERFACE"]:
                    context_knowledge += f"- {node.content} (相关度: {score:.2f})\n"
        
        # 获取步骤数据
        steps = data.get("steps", [])
        steps_text = ""
        
        for i, step in enumerate(steps):
            if isinstance(step, dict) and "description" in step:
                steps_text += f"步骤{i+1}: {step['description']}\n"
                if "action" in step:
                    steps_text += f"  - 操作: {step['action']}\n"
                if "target" in step:
                    steps_text += f"  - 目标: {step['target']}\n"
            else:
                steps_text += f"步骤{i+1}: {str(step)}\n"
        
        if not steps_text:
            return {
                "score": 0.2,
                "feedback": ["缺少操作步骤"],
                "suggestions": ["添加具体操作步骤"]
            }
        
        # 使用LLM进行综合评估
        prompt = f"""
请作为经验评估专家，评估以下"HOW"类型的操作步骤质量。

步骤内容:
{steps_text}

相关操作知识:
{context_knowledge if context_knowledge else "无相关操作知识"}

请从以下维度进行评估(1-10分):
1. 完整性: 步骤是否涵盖了完整流程
2. 精确性: 操作描述是否精确
3. 逻辑流程: 步骤顺序是否合理
4. 边缘情况处理: 是否考虑了异常情况
5. 效率: 操作流程是否高效

请以JSON格式返回评估结果:
{{
  "scores": {{
    "completeness": 评分,
    "precision": 评分,
    "logical_flow": 评分,
    "edge_case_handling": 评分,
    "efficiency": 评分
  }},
  "overall_score": 0-1之间的总体评分,
  "feedback": ["反馈1", "反馈2", ...],
  "suggestions": ["建议1", "建议2", ...]
}}
"""
        
        try:
            response = call_openai(prompt)
            result = json.loads(response)
            
            if "scores" in result:
                scores = result["scores"]
            
            if "feedback" in result and isinstance(result["feedback"], list):
                feedback = result["feedback"]
                
            if "suggestions" in result and isinstance(result["suggestions"], list):
                suggestions = result["suggestions"]
                
            # 如果LLM返回了整体评分，使用它
            if "overall_score" in result:
                overall_score = float(result["overall_score"])
                # 确保分数在0-1范围内
                overall_score = max(0.0, min(1.0, overall_score))
            else:
                # 否则计算加权平均分
                overall_score = sum(
                    scores.get(criterion, 5) / 10 * weight 
                    for criterion, weight in criteria.items()
                )
            
            return {
                "score": overall_score,
                "detail_scores": scores,
                "feedback": feedback,
                "suggestions": suggestions
            }
            
        except Exception as e:
            logger.error(f"HOW片段评估失败: {str(e)}")
            # 备用评估逻辑
            steps_score = min(1.0, len(steps) / 5)  # 假设5步是理想长度
            return {
                "score": steps_score * 0.6 + 0.2,
                "feedback": [f"自动评估: 包含{len(steps)}个步骤"],
                "suggestions": ["增加步骤详细描述"]
            }
    
    def _evaluate_check_fragment(self, data: Dict, related_nodes: List, interactive: bool) -> Dict:
        """评估CHECK类型片段"""
        if not data:
            return {"score": 0.1, "feedback": ["CHECK片段数据为空"]}
            
        criteria = self.evaluation_criteria["CHECK"]
        scores = {}
        feedback = []
        suggestions = []
        
        # 从知识图谱中提取相关验证知识
        context_knowledge = ""
        if related_nodes:
            for node, score in related_nodes[:3]:  # 取前3个最相关的节点
                if node.type in ["RULE", "VALIDATION", "TEST", "CRITERIA"]:
                    context_knowledge += f"- {node.content} (相关度: {score:.2f})\n"
        
        # 获取规则数据
        rules = data.get("rules", [])
        rules_text = "\n".join([f"- {rule}" for rule in rules])
        
        if not rules_text:
            return {
                "score": 0.2,
                "feedback": ["缺少验证规则"],
                "suggestions": ["添加具体验证规则"]
            }
        
        # 使用LLM进行综合评估
        prompt = f"""
请作为经验评估专家，评估以下"CHECK"类型的验证规则质量。

规则内容:
{rules_text}

相关验证知识:
{context_knowledge if context_knowledge else "无相关验证知识"}

请从以下维度进行评估(1-10分):
1. 覆盖面: 规则是否全面覆盖验证需求
2. 具体性: 规则是否具体明确
3. 可测试性: 规则是否可验证/测试
4. 错误处理: 是否考虑了异常情况

请以JSON格式返回评估结果:
{{
  "scores": {{
    "coverage": 评分,
    "specificity": 评分,
    "testability": 评分,
    "error_handling": 评分
  }},
  "overall_score": 0-1之间的总体评分,
  "feedback": ["反馈1", "反馈2", ...],
  "suggestions": ["建议1", "建议2", ...]
}}
"""
        
        try:
            response = call_openai(prompt)
            result = json.loads(response)
            
            if "scores" in result:
                scores = result["scores"]
            
            if "feedback" in result and isinstance(result["feedback"], list):
                feedback = result["feedback"]
                
            if "suggestions" in result and isinstance(result["suggestions"], list):
                suggestions = result["suggestions"]
                
            # 如果LLM返回了整体评分，使用它
            if "overall_score" in result:
                overall_score = float(result["overall_score"])
                # 确保分数在0-1范围内
                overall_score = max(0.0, min(1.0, overall_score))
            else:
                # 否则计算加权平均分
                overall_score = sum(
                    scores.get(criterion, 5) / 10 * weight 
                    for criterion, weight in criteria.items()
                )
            
            return {
                "score": overall_score,
                "detail_scores": scores,
                "feedback": feedback,
                "suggestions": suggestions
            }
            
        except Exception as e:
            logger.error(f"CHECK片段评估失败: {str(e)}")
            # 备用评估逻辑
            rules_score = min(1.0, len(rules) / 3)  # 假设3个规则是理想数量
            return {
                "score": rules_score * 0.6 + 0.2,
                "feedback": [f"自动评估: 包含{len(rules)}条规则"],
                "suggestions": ["增加规则详细描述"]
            }
    
    def _evaluate_generic_fragment(self, data: Dict, related_nodes: List, interactive: bool) -> Dict:
        """评估通用类型片段"""
        if not data:
            return {"score": 0.1, "feedback": ["片段数据为空"]}
        
        # 数据完整性分析
        completeness = self._calculate_completeness(data)
        
        # 内容分析
        content_text = json.dumps(data, ensure_ascii=False)
        
        # 使用LLM进行通用评估
        prompt = f"""
请作为经验评估专家，评估以下经验片段的质量。

片段内容:
{content_text}

请评估此片段的质量，并从以下维度打分(1-10分):
1. 内容完整性
2. 相关性
3. 实用性

请以JSON格式返回评估结果:
{{
  "scores": {{
    "completeness": 评分,
    "relevance": 评分,
    "usefulness": 评分
  }},
  "overall_score": 0-1之间的总体评分,
  "feedback": ["反馈1", "反馈2"],
  "suggestions": ["建议1", "建议2"]
}}
"""
        
        try:
            response = call_openai(prompt)
            result = json.loads(response)
            
            # 提取评分和反馈
            scores = result.get("scores", {})
            feedback = result.get("feedback", [])
            suggestions = result.get("suggestions", [])
            
            # 计算总分
            if "overall_score" in result:
                overall_score = float(result["overall_score"])
                # 确保分数在0-1范围内
                overall_score = max(0.0, min(1.0, overall_score))
            else:
                avg_score = sum(scores.values()) / len(scores) if scores else 5
                overall_score = avg_score / 10
                
            return {
                "score": overall_score,
                "detail_scores": scores,
                "feedback": feedback,
                "suggestions": suggestions
            }
            
        except Exception as e:
            logger.error(f"通用片段评估失败: {str(e)}")
            # 备用评估逻辑
            return {
                "score": completeness * 0.6 + 0.2,
                "feedback": [f"自动评估: 数据完整度 {completeness*100:.0f}%"],
                "suggestions": ["完善片段数据"]
            }
    
    def evaluate_experience_pack(self, exp_pack, interactive: bool = False) -> Dict[str, Any]:
        """
        评估整个经验包
        
        Args:
            exp_pack: 经验包对象
            interactive: 是否启用交互式评估
            
        Returns:
            整体评估结果
        """
        if not hasattr(exp_pack, "fragments") or not exp_pack.fragments:
            return {
                "overall_score": 0.1,
                "message": "经验包为空",
                "suggestions": ["添加经验片段"]
            }
            
        fragment_results = {}
        overall_suggestions = []
        
        # 评估每个片段
        for fragment in exp_pack.fragments:
            result = self.evaluate_fragment(fragment, interactive)
            fragment_results[fragment.frag_type] = result
            
            # 收集改进建议
            if "suggestions" in result:
                for suggestion in result["suggestions"]:
                    overall_suggestions.append(f"{fragment.frag_type}: {suggestion}")
        
        # 计算总体评分
        type_weights = {"WHY": 0.4, "HOW": 0.4, "CHECK": 0.2}
        weighted_sum = 0.0
        total_weight = 0.0
        
        for frag_type, result in fragment_results.items():
            weight = type_weights.get(frag_type, 0.1)
            weighted_sum += result.get("score", 0) * weight
            total_weight += weight
            
        overall_score = weighted_sum / total_weight if total_weight > 0 else 0
        
        # 检查是否缺少重要片段类型
        missing_types = []
        for key_type in ["WHY", "HOW", "CHECK"]:
            if key_type not in fragment_results:
                missing_types.append(key_type)
                overall_suggestions.insert(0, f"缺少{key_type}类型片段")
        
        # 生成整体评估总结
        if exp_pack.kg and hasattr(exp_pack.kg, "get_all_nodes"):
            kg_size = len(exp_pack.kg.get_all_nodes())
        else:
            kg_size = 0
            
        completeness_score = (3 - len(missing_types)) / 3
        
        # 使用LLM进行整体评估总结
        if fragment_results:
            try:
                fragments_summary = []
                for ftype, res in fragment_results.items():
                    fragments_summary.append({
                        "type": ftype,
                        "score": res.get("score", 0),
                        "key_feedback": res.get("feedback", [])[:2]  # 最多取两条反馈
                    })
                    
                summary_prompt = f"""
请作为经验评估专家，根据以下经验片段的评估结果，生成整体评估总结:

任务名称: {exp_pack.task_name}
知识点数量: {kg_size}
完整度: {completeness_score:.2f} (缺少类型: {', '.join(missing_types) if missing_types else "无"})

片段评估结果:
{json.dumps(fragments_summary, ensure_ascii=False, indent=2)}

请生成一个简短的总结评估，以及3-5条具体的改进建议。

格式:
{{
  "summary": "总体评价...",
  "quality_level": "高/中/低",
  "key_strengths": ["优势1", "优势2"],
  "improvement_suggestions": ["建议1", "建议2", "建议3"]
}}
"""
                response = call_openai(summary_prompt)
                summary_result = json.loads(response)
                
                # 整合最终结果
                return {
                    "overall_score": overall_score,
                    "fragment_scores": fragment_results,
                    "quality_level": summary_result.get("quality_level", "中"),
                    "summary": summary_result.get("summary", ""),
                    "key_strengths": summary_result.get("key_strengths", []),
                    "suggestions": summary_result.get("improvement_suggestions", overall_suggestions[:5])
                }
                
            except Exception as e:
                logger.error(f"生成整体评估失败: {str(e)}")
                
        # 备用返回值
        return {
            "overall_score": overall_score,
            "fragment_scores": fragment_results,
            "quality_level": "高" if overall_score > 0.8 else "中" if overall_score > 0.5 else "低",
            "completeness": completeness_score,
            "suggestions": overall_suggestions[:5]  # 最多返回5条建议
        }
    
    def _calculate_completeness(self, data: Dict) -> float:
        """计算数据完整度"""
        if not data:
            return 0.0
            
        filled_fields = 0
        total_fields = 0
        
        for key, value in data.items():
            total_fields += 1
            
            if value is not None:
                if isinstance(value, str) and value.strip():
                    filled_fields += 1
                elif isinstance(value, (list, dict)) and len(value) > 0:
                    filled_fields += 1
                elif isinstance(value, (int, float, bool)):
                    filled_fields += 1
                    
        return filled_fields / total_fields if total_fields > 0 else 0.0


# 示例用法 - 修复测试部分
if __name__ == "__main__":
    # 不再尝试直接操作知识图谱，而是使用模拟对象进行测试
    
    # 模拟知识图谱节点
    class MockKnowledgePoint:
        def __init__(self, node_id, node_type, content):
            self.id = node_id
            self.type = node_type
            self.content = content
    
    # 模拟知识图谱
    class MockExperienceGraph:
        def __init__(self, name):
            self.name = name
            self._nodes = []
            
        def get_all_nodes(self):
            return self._nodes
            
        # 添加模拟节点供测试使用
        def add_mock_nodes(self, nodes):
            self._nodes.extend(nodes)
    
    # 创建模拟知识图谱和节点
    mock_kg = MockExperienceGraph("test_graph")
    mock_kg.add_mock_nodes([
        MockKnowledgePoint("1", "GOAL", "自动验证页面结构变化"),
        MockKnowledgePoint("2", "BACKGROUND", "页面频繁变化导致测试成本高"),
        MockKnowledgePoint("3", "OPERATION", "截取页面快照"),
        MockKnowledgePoint("4", "RULE", "验证页面元素位置变化不超过10px")
    ])
    
    # 创建评估器，使用模拟知识图谱
    evaluator = ExperienceEvaluator(mock_kg)
    
    # 模拟fragment进行测试
    class MockFragment:
        def __init__(self, frag_type, data):
            self.frag_type = frag_type
            self.data = data
    
    why_fragment = MockFragment("WHY", {
        "goal": "自动验证页面结构变化",
        "background": "页面变化频繁，测试覆盖不足",
        "constraints": ["不能改代码", "支持多端"],
        "expected_outcome": "自动生成验证用例"
    })
    
    # 测试评估
    result = evaluator.evaluate_fragment(why_fragment)
    print(json.dumps(result, indent=2, ensure_ascii=False))