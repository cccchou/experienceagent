# experienceagent/fragment_recommender.py
"""
经验片段推荐系统
适配rich_expert_validation.json格式
添加GPT生成补充功能，当经验库没有相关内容时自动生成
"""

from typing import List, Dict, Any, Optional, Tuple, Union
from openai import OpenAI
import json
import os
import logging
import time
import re
from collections import defaultdict
from experienceagent.fragment_scorer import ExperienceEvaluator

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ExperienceSystem")

client = OpenAI(
)

def call_openai(prompt: str, system_prompt: str = None, model: str = "deepseek-chat") -> str:
    """调用OpenAI API获取结果"""
    if system_prompt is None:
        system_prompt = "你是一个经验推荐专家，请根据用户需求推荐最相关的经验片段。"
        
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content


def extract_json_from_text(text: str) -> str:
    """尝试从文本中提取JSON部分"""
    # 尝试查找JSON数组格式，通常是 [ 开头，] 结尾
    json_array_match = re.search(r'\[(.*?)\]', text, re.DOTALL)
    if json_array_match:
        potential_json = f"[{json_array_match.group(1)}]"
        try:
            # 验证是否是有效的JSON
            json.loads(potential_json)
            return potential_json
        except:
            pass
    
    # 尝试查找JSON对象格式，通常是 { 开头，} 结尾
    json_object_match = re.search(r'\{(.*?)\}', text, re.DOTALL)
    if json_object_match:
        potential_json = f"{{{json_object_match.group(1)}}}"
        try:
            # 验证是否是有效的JSON
            json.loads(potential_json)
            return potential_json
        except:
            pass
    
    # 尝试查找整个文本中的JSON对象
    try:
        # 查找可能的JSON起始
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        
        if start_idx != -1 and end_idx != -1:
            potential_json = text[start_idx:end_idx+1]
            # 验证是否是有效的JSON
            json.loads(potential_json)
            return potential_json
    except:
        pass
    
    # 查找数组格式
    try:
        start_idx = text.find("[")
        end_idx = text.rfind("]")
        
        if start_idx != -1 and end_idx != -1:
            potential_json = text[start_idx:end_idx+1]
            # 验证是否是有效的JSON
            json.loads(potential_json)
            return potential_json
    except:
        pass
    
    # 如果没有找到有效的JSON，返回原始文本
    return text


# 定义经验类型映射
EXPERIENCE_TYPE_MAPPING = {
    "why_structured": "WHY",
    "how_behavior_logs": "HOW", 
    "check_rules": "CHECK",
    "dialogue_logs": "DIALOGUE"
}


class ExperienceRetriever:
    """
    经验检索器
    适配rich_expert_validation.json格式
    """
    def __init__(self, db_path: str, evaluator: ExperienceEvaluator = None):
        """
        初始化经验检索器
        
        Args:
            db_path: 经验数据库路径
            evaluator: 评分器实例
        """
        self.db_path = db_path
        self.evaluator = evaluator
        
        # 经验库
        self.experience_db = []
        
        # 索引 - 加速检索
        self.type_index = defaultdict(list)  # 按经验类型索引
        self.tag_index = defaultdict(list)   # 按标签索引
        
        # 加载数据库
        if db_path and os.path.exists(db_path):
            self._load_db()
    
    def _load_db(self) -> bool:
        """加载经验数据库"""
        if not self.db_path or not os.path.exists(self.db_path):
            logger.warning(f"数据库文件不存在: {self.db_path}")
            return False
            
        try:
            with open(self.db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 处理不同格式的数据
            if isinstance(data, dict) and "experiences" in data:
                # 包含多个字段的完整格式
                self.experience_db = data.get("experiences", [])
            elif isinstance(data, dict) and any(key in EXPERIENCE_TYPE_MAPPING for key in data.keys()):
                # 单个经验的rich_expert_validation格式
                self.experience_db = [data]
            elif isinstance(data, list):
                # 经验列表格式
                self.experience_db = data
            else:
                # 尝试作为单个经验处理
                self.experience_db = [data]
            
            # 构建索引
            self._build_indexes()
            logger.info(f"已加载 {len(self.experience_db)} 条经验")
            return True
        except Exception as e:
            logger.error(f"加载经验数据库失败: {str(e)}")
            return False
    
    def _build_indexes(self):
        """构建检索索引 - 适配rich_expert_validation.json格式"""
        # 清空现有索引
        self.type_index = defaultdict(list)
        self.tag_index = defaultdict(list)
        
        # 重建索引
        for idx, exp in enumerate(self.experience_db):
            # 经验类型索引 - 直接使用字段名映射
            for field_name, type_name in EXPERIENCE_TYPE_MAPPING.items():
                if field_name in exp:
                    self.type_index[type_name].append(idx)
            
            # 兼容fragments格式
            if "fragments" in exp:
                for frag in exp["fragments"]:
                    frag_type = frag.get("type")
                    if frag_type:
                        self.type_index[frag_type].append(idx)
            
            # 标签索引 (如果有)
            if "tags" in exp:
                for tag in exp["tags"]:
                    self.tag_index[tag].append(idx)
    
    def save_db(self, path: str = None) -> bool:
        """
        保存经验数据库
        
        Args:
            path: 保存路径
            
        Returns:
            是否保存成功
        """
        save_path = path or self.db_path
        if not save_path:
            logger.warning("未指定保存路径")
            return False
            
        try:
            # 保持原有格式
            with open(save_path, 'w', encoding='utf-8') as f:
                if len(self.experience_db) == 1:
                    # 单个经验格式
                    json.dump(self.experience_db[0], f, ensure_ascii=False, indent=2)
                else:
                    # 经验列表格式
                    json.dump(self.experience_db, f, ensure_ascii=False, indent=2)
                
            logger.info(f"已保存 {len(self.experience_db)} 条经验到 {save_path}")
            return True
        except Exception as e:
            logger.error(f"保存经验数据库失败: {str(e)}")
            return False
    
    def add_experience(self, experience_pack, context: str = None) -> Dict:
        """
        添加经验到数据库，适配rich_expert_validation.json格式
        
        Args:
            experience_pack: 经验包对象
            context: 对话上下文 (可选)
            
        Returns:
            添加结果
        """
        # 转换为rich_expert_validation格式
        exp_data = {
            "task_title": experience_pack.task_name  # 添加任务标题，保持一致性
        }
        
        # 添加对话日志（如果有）
        if context:
            exp_data["dialogue_logs"] = [
                f"用户: {context}",
                f"系统: 已记录关于「{experience_pack.task_name}」的经验"
            ]
        
        # 转换片段为对应格式
        for fragment in experience_pack.fragments:
            if fragment.frag_type == "WHY":
                exp_data["why_structured"] = fragment.data
            elif fragment.frag_type == "HOW":
                # 转换HOW片段为how_behavior_logs格式
                if "steps" in fragment.data:
                    behavior_logs = []
                    for step in fragment.data["steps"]:
                        if isinstance(step, dict):
                            behavior_log = {
                                "page": step.get("page", "未指定页面"),
                                "action": step.get("action", "执行"),
                                "element": step.get("element", "未指定元素"),
                                "intent": step.get("description", step.get("intent", "未指定意图"))
                            }
                            behavior_logs.append(behavior_log)
                    exp_data["how_behavior_logs"] = behavior_logs
            elif fragment.frag_type == "CHECK":
                if "rules" in fragment.data:
                    exp_data["check_rules"] = fragment.data["rules"]
        
        # 添加到数据库
        self.experience_db.append(exp_data)
        
        # 更新索引
        self._build_indexes()
        
        # 保存更新后的数据库
        if self.db_path:
            self.save_db()
            
        return {
            "success": True,
            "index": len(self.experience_db) - 1
        }
    
    def retrieve_by_fragment_type(self, frag_type: str) -> List[Dict]:
        """
        按片段类型检索经验
        
        Args:
            frag_type: 片段类型 (WHY, HOW, CHECK)
            
        Returns:
            匹配经验列表
        """
        # 使用索引查找
        matching_indexes = self.type_index.get(frag_type, [])
        results = []
        
        # 字段映射 - 从标准类型到rich_expert_validation字段
        field_mapping = {v: k for k, v in EXPERIENCE_TYPE_MAPPING.items()}
        field_name = field_mapping.get(frag_type)
        
        for idx in matching_indexes:
            exp = self.experience_db[idx]
            
            # 构建结果（适配rich_expert_validation格式）
            if field_name and field_name in exp:
                # 为保持API兼容性，我们构建一个fragments结构
                fragment = {
                    "type": frag_type,
                    "data": exp[field_name]
                }
                
                result = {
                    "task_name": exp.get("task_title", "未命名任务"),
                    "fragments": [fragment]
                }
                results.append(result)
                
            # 兼容fragments格式
            elif "fragments" in exp:
                fragments = []
                for frag in exp["fragments"]:
                    if frag.get("type") == frag_type:
                        fragments.append(frag)
                        
                if fragments:
                    result = {
                        "task_name": exp.get("task_name", "未命名任务"),
                        "fragments": fragments
                    }
                    results.append(result)
                    
        return results
    
    def semantic_search(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """
        语义搜索经验
        
        Args:
            query_text: 查询文本
            top_k: 返回结果数量
            
        Returns:
            相关经验列表
        """
        # 准备候选项
        candidates = []
        for idx, exp in enumerate(self.experience_db):
            # 提取文本表示
            text_repr = self._get_experience_text(exp)
            candidates.append({
                "id": idx,
                "text": text_repr
            })
        
        if not candidates:
            return []
            
        try:
            # 使用OpenAI进行语义匹配
            matches = self._semantic_matching(query_text, candidates, top_k)
            
            # 返回匹配结果
            results = []
            for match in matches:
                idx = match.get("id")
                if 0 <= idx < len(self.experience_db):
                    results.append({
                        "experience": self.experience_db[idx],
                        "similarity": match.get("similarity", 0),
                        "reason": match.get("reason", "")
                    })
                    
            return results
            
        except Exception as e:
            logger.error(f"语义搜索失败: {str(e)}")
            # 备用方案 - 简单文本匹配
            return self._fallback_text_search(query_text, candidates, top_k)
    
    def _semantic_matching(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        """使用OpenAI进行语义匹配"""
        # 准备候选项
        candidate_texts = []
        for i, candidate in enumerate(candidates):
            # 限制文本长度
            text = candidate["text"]
            if len(text) > 500:  # 限制长度，避免过长
                text = text[:500] + "..."
                
            candidate_texts.append({
                "id": candidate["id"],
                "text": text
            })
        
        # 构建提示语 - 明确强调JSON格式要求
        prompt = f"""
请作为专业的经验检索专家，从候选项中找出与查询最相关的{top_k}个经验:

查询: {query}

候选经验:
{json.dumps(candidate_texts, ensure_ascii=False)}

请返回严格的JSON格式结果，不要添加任何额外说明或标记，只返回一个包含以下结构的JSON数组:
[
  {{"id": 经验ID, "similarity": 相关性评分, "reason": "匹配原因简述"}},
  ...
]

相关性评分必须是0-1之间的数字。按相关性从高到低排序，只返回最相关的{top_k}个结果。
"""
        
        try:
            # 调用模型
            response = call_openai(prompt)
            
            # 处理模型响应
            logger.debug(f"原始模型响应: {response}")
            
            # 尝试从响应中提取有效JSON
            json_str = extract_json_from_text(response)
            
            try:
                # 尝试解析JSON
                result = json.loads(json_str)
            except json.JSONDecodeError as je:
                logger.warning(f"JSON解析失败: {str(je)}，尝试使用备用方法")
                # 手动构建结果 - 使用简单启发式方法
                result = self._parse_results_manually(response, candidates, top_k)
                
            # 确保结果是一个列表
            if not isinstance(result, list):
                logger.warning(f"模型返回的不是一个列表，而是: {type(result)}")
                result = []
                
            # 确保每个项目有必要的字段
            valid_results = []
            for item in result:
                if isinstance(item, dict) and "id" in item:
                    # 确保similarity是浮点数
                    if "similarity" not in item or not isinstance(item["similarity"], (int, float)):
                        item["similarity"] = 0.5  # 默认相似度
                    else:
                        # 限制范围在0-1之间
                        item["similarity"] = max(0.0, min(1.0, float(item["similarity"])))
                    
                    # 确保有reason字段
                    if "reason" not in item:
                        item["reason"] = "相关内容匹配"
                        
                    valid_results.append(item)
                    
            # 如果有效结果为空，使用备用方法
            if not valid_results:
                logger.warning("有效匹配结果为空，使用备用方法")
                valid_results = self._parse_results_manually(response, candidates, top_k)
            
            # 排序并返回结果
            valid_results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
            return valid_results[:top_k]
            
        except Exception as e:
            logger.error(f"语义匹配过程中发生错误: {str(e)}")
            return []
    
    def _parse_results_manually(self, response: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        """手动解析模型响应，提取可能的匹配结果"""
        results = []
        
        # 尝试从文本中提取ID和相关性
        # 寻找如 "ID: 1, 相似度: 0.8" 这样的模式
        id_pattern = r"(?:ID|id)[^\d]*(\d+)"
        similarity_pattern = r"(?:相似度|similarity|相关性)[^\d]*([0-9]+\.?[0-9]*)"
        
        # 按行分割响应
        lines = response.strip().split('\n')
        
        current_id = None
        current_similarity = None
        current_reason = ""
        
        for line in lines:
            line = line.strip()
            
            # 检查是否提到了ID
            id_match = re.search(id_pattern, line)
            if id_match:
                # 如果之前有解析到一个完整条目，先添加它
                if current_id is not None:
                    results.append({
                        "id": current_id,
                        "similarity": current_similarity or 0.5,
                        "reason": current_reason or "相关内容匹配"
                    })
                
                # 开始新的条目
                current_id = int(id_match.group(1))
                current_similarity = None
                current_reason = ""
                
            # 检查是否提到了相似度
            similarity_match = re.search(similarity_pattern, line)
            if similarity_match and current_id is not None:
                try:
                    current_similarity = float(similarity_match.group(1))
                    # 确保在0-1范围内
                    current_similarity = max(0.0, min(1.0, current_similarity))
                except:
                    current_similarity = 0.5
            
            # 累积reason (只保留第一行作为reason)
            if current_id is not None and not current_reason and not id_match and not similarity_match and line:
                current_reason = line
        
        # 添加最后一个条目
        if current_id is not None:
            results.append({
                "id": current_id,
                "similarity": current_similarity or 0.5,
                "reason": current_reason or "相关内容匹配"
            })
        
        # 如果仍然没有结果，尝试按候选项顺序创建默认结果
        if not results:
            for i, candidate in enumerate(candidates[:top_k]):
                results.append({
                    "id": candidate["id"],
                    "similarity": 0.5 * (top_k - i) / top_k,  # 简单的递减相似度
                    "reason": "默认匹配"
                })
                
        return results
    
    def _fallback_text_search(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        """简单文本匹配的备用搜索"""
        results = []
        query_terms = query.lower().split()
        
        for candidate in candidates:
            text = candidate["text"].lower()
            # 简单计算匹配次数
            match_count = sum(1 for term in query_terms if term in text)
            if match_count > 0:
                similarity = min(1.0, match_count / len(query_terms))
                results.append({
                    "experience": self.experience_db[candidate["id"]],
                    "similarity": similarity,
                    "reason": "文本匹配"
                })
                
        # 排序并限制结果数
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]
    
    def _get_experience_text(self, experience: Dict) -> str:
        """获取经验的文本表示 - 适配rich_expert_validation.json格式"""
        task_name = experience.get("task_title", "")
        if not task_name:
            task_name = experience.get("task_name", "未命名任务")
            
        text = f"任务: {task_name}\n"
        
        # 添加WHY内容
        if "why_structured" in experience:
            why_data = experience["why_structured"]
            text += "\n--- WHY ---\n"
            if "goal" in why_data:
                text += f"目标: {why_data['goal']}\n"
            if "background" in why_data:
                text += f"背景: {why_data['background']}\n"
            if "constraints" in why_data and isinstance(why_data["constraints"], list):
                text += f"约束: {', '.join(why_data['constraints'])}\n"
            if "expected_outcome" in why_data:
                text += f"预期效果: {why_data['expected_outcome']}\n"
                
        # 添加HOW内容
        if "how_behavior_logs" in experience:
            logs = experience["how_behavior_logs"]
            text += "\n--- HOW ---\n"
            for i, log in enumerate(logs):
                text += f"步骤{i+1}: "
                if isinstance(log, dict):
                    text += f"{log.get('action', '')} {log.get('element', '')} "
                    text += f"在 {log.get('page', '')} 页面，目的是 {log.get('intent', '')}\n"
                else:
                    text += f"{str(log)}\n"
                    
        # 添加CHECK内容
        if "check_rules" in experience:
            rules = experience["check_rules"]
            text += "\n--- CHECK ---\n"
            for i, rule in enumerate(rules):
                text += f"规则{i+1}: {rule}\n"
                
        # 添加对话日志
        if "dialogue_logs" in experience:
            logs = experience["dialogue_logs"]
            text += "\n--- 对话日志 ---\n"
            for log in logs:
                text += f"{log}\n"
                
        # 兼容fragments格式
        if "fragments" in experience:
            text += "\n--- 经验片段 ---\n"
            for fragment in experience["fragments"]:
                frag_type = fragment.get("type", "")
                frag_data = fragment.get("data", {})
                
                text += f"\n--- {frag_type} ---\n"
                text += json.dumps(frag_data, ensure_ascii=False)
                
        # 添加标签 (如果有)
        if "tags" in experience:
            tags = experience["tags"]
            if tags:
                text += f"\n标签: {', '.join(tags)}\n"
                
        return text
    
    def recommend_complementary_fragments(self, fragments: List, top_k: int = 2) -> Dict[str, List]:
        """
        推荐互补的经验片段
        
        Args:
            fragments: 已有片段列表
            top_k: 每种类型返回的推荐数量
            
        Returns:
            按类型分类的推荐片段
        """
        # 分析现有片段类型
        existing_types = set(frag.frag_type for frag in fragments if hasattr(frag, "frag_type"))
        
        # 推荐缺失的片段类型
        recommendations = {}
        for frag_type in ["WHY", "HOW", "CHECK"]:
            if frag_type not in existing_types:
                results = self.retrieve_by_fragment_type(frag_type)
                
                # 限制推荐数量
                if results:
                    recommendations[frag_type] = results[:top_k]
                    
        return recommendations
    
    def get_stats(self) -> Dict[str, Any]:
        """获取经验库统计信息"""
        # 片段类型分布
        type_counts = {}
        for frag_type, indexes in self.type_index.items():
            type_counts[frag_type] = len(indexes)
            
        return {
            "total": len(self.experience_db),
            "fragment_types": type_counts
        }


class FragmentRecommender:
    """
    经验片段推荐器
    基于经验库提供智能推荐功能，并记录客户交互
    增加AI生成推荐功能
    """
    def __init__(self, retriever: ExperienceRetriever = None, evaluator: ExperienceEvaluator = None,
                db_path: str = None, min_recommendations: int = 2, min_similarity: float = 0.4):
        """
        初始化推荐器
        
        Args:
            retriever: 经验检索器
            evaluator: 评分器
            db_path: 数据库路径
            min_recommendations: 最少推荐数量
            min_similarity: 最低相似度阈值
        """
        # 优先使用传入的检索器
        if retriever:
            self.retriever = retriever
        elif db_path:
            self.retriever = ExperienceRetriever(db_path, evaluator)
        else:
            raise ValueError("必须提供检索器或数据库路径")
            
        self.evaluator = evaluator
        self.dialogue_history = []  # 记录对话历史
        
        # 推荐阈值设置
        self.min_recommendations = min_recommendations  # 每种类型最少推荐数量
        self.min_similarity = min_similarity  # 最低相似度阈值
    
    def recommend_for_task(self, task_description: str, existing_fragments: List = None, 
                          dialog_context: str = None) -> Dict[str, List]:
        """
        基于任务描述推荐经验片段
        如果经验库中没有足够相关内容，自动生成推荐
        
        Args:
            task_description: 任务描述
            existing_fragments: 已有片段列表
            dialog_context: 对话上下文
            
        Returns:
            推荐结果
        """
        # 记录交互
        if dialog_context:
            self.dialogue_history.append(f"用户: {dialog_context}")
            self.dialogue_history.append(f"系统: 为任务「{task_description}」提供推荐")
        
        # 查找相似经验
        similar_experiences = self.retriever.semantic_search(
            task_description, top_k=5
        )
        
        # 分析已有片段类型
        existing_types = set()
        if existing_fragments:
            existing_types = set(frag.frag_type for frag in existing_fragments if hasattr(frag, "frag_type"))
            
        # 提取全面推荐 - 按片段类型组织
        recommendations = {}
        fragment_sources = {}  # 追踪片段来源
        
        for item in similar_experiences:
            exp = item["experience"]
            similarity = item["similarity"]
            
            # 跳过低相似度的结果
            if similarity < self.min_similarity:
                continue
                
            # 获取任务名称
            task_name = exp.get("task_title", "")
            if not task_name:
                task_name = exp.get("task_name", "未命名任务")
                
            # 处理rich_expert_validation.json格式
            # 检查WHY类型
            if "why_structured" in exp and "WHY" not in existing_types:
                if "WHY" not in recommendations:
                    recommendations["WHY"] = []
                    fragment_sources["WHY"] = set()
                    
                if task_name not in fragment_sources["WHY"]:
                    recommendations["WHY"].append({
                        "fragment": {
                            "type": "WHY",
                            "data": exp["why_structured"]
                        },
                        "task_name": task_name,
                        "similarity": similarity,
                        "source": "database"  # 标记来源为数据库
                    })
                    fragment_sources["WHY"].add(task_name)
                    
            # 检查HOW类型
            if "how_behavior_logs" in exp and "HOW" not in existing_types:
                if "HOW" not in recommendations:
                    recommendations["HOW"] = []
                    fragment_sources["HOW"] = set()
                    
                if task_name not in fragment_sources["HOW"]:
                    recommendations["HOW"].append({
                        "fragment": {
                            "type": "HOW",
                            "data": {"steps": exp["how_behavior_logs"]}
                        },
                        "task_name": task_name,
                        "similarity": similarity,
                        "source": "database"  # 标记来源为数据库
                    })
                    fragment_sources["HOW"].add(task_name)
                    
            # 检查CHECK类型
            if "check_rules" in exp and "CHECK" not in existing_types:
                if "CHECK" not in recommendations:
                    recommendations["CHECK"] = []
                    fragment_sources["CHECK"] = set()
                    
                if task_name not in fragment_sources["CHECK"]:
                    recommendations["CHECK"].append({
                        "fragment": {
                            "type": "CHECK",
                            "data": {"rules": exp["check_rules"]}
                        },
                        "task_name": task_name,
                        "similarity": similarity,
                        "source": "database"  # 标记来源为数据库
                    })
                    fragment_sources["CHECK"].add(task_name)
                    
            # 兼容fragments格式
            if "fragments" in exp:
                for fragment in exp["fragments"]:
                    frag_type = fragment.get("type")
                    
                    # 跳过已有类型
                    if frag_type in existing_types:
                        continue
                        
                    # 创建类型条目
                    if frag_type not in recommendations:
                        recommendations[frag_type] = []
                        fragment_sources[frag_type] = set()
                        
                    # 避免重复推荐相同经验中的片段
                    if task_name not in fragment_sources[frag_type]:
                        recommendations[frag_type].append({
                            "fragment": fragment,
                            "task_name": task_name,
                            "similarity": similarity,
                            "source": "database"  # 标记来源为数据库
                        })
                        fragment_sources[frag_type].add(task_name)
        
        # 对每种类型的推荐结果排序
        for frag_type in recommendations:
            recommendations[frag_type].sort(
                key=lambda x: x["similarity"],
                reverse=True
            )
            
        # 检查是否需要AI生成补充推荐
        needed_types = ["WHY", "HOW", "CHECK"]
        for frag_type in needed_types:
            # 跳过已有的片段类型
            if frag_type in existing_types:
                continue
                
            # 如果某种类型的推荐不足，使用AI生成补充
            if frag_type not in recommendations or len(recommendations[frag_type]) < self.min_recommendations:
                generated_fragments = self._generate_fragment(task_description, frag_type, dialog_context)
                
                if generated_fragments:
                    # 确保推荐列表已初始化
                    if frag_type not in recommendations:
                        recommendations[frag_type] = []
                    
                    # 添加生成的片段
                    for gen_fragment in generated_fragments:
                        recommendations[frag_type].append(gen_fragment)
                    
                    # 重新排序
                    recommendations[frag_type].sort(
                        key=lambda x: x["similarity"],
                        reverse=True
                    )
                    
                    # 保存生成的片段到经验库
                    for gen_fragment in generated_fragments:
                        # 创建一个模拟经验包
                        from experienceagent.controller_agent import ExperiencePack, ExperienceFragment
                        exp_pack = ExperiencePack(f"AI生成: {task_description}")
                        exp_fragment = ExperienceFragment(frag_type, gen_fragment["fragment"]["data"])
                        exp_pack.add_fragment(exp_fragment)
                        
                        # 添加到经验库
                        context = f"AI为查询「{task_description}」生成的{frag_type}片段"
                        if dialog_context:
                            context = f"{dialog_context} -> {context}"
                            
                        self.retriever.add_experience(exp_pack, context)
                        logger.info(f"已将AI生成的{frag_type}片段添加到经验库")
            
        return recommendations
    
    def _generate_fragment(self, task_description: str, frag_type: str, 
                          dialog_context: str = None) -> List[Dict]:
        """
        使用AI生成特定类型的片段
        
        Args:
            task_description: 任务描述
            frag_type: 片段类型
            dialog_context: 对话上下文
            
        Returns:
            生成的片段列表
        """
        logger.info(f"为任务「{task_description}」生成{frag_type}片段")
        
        # 构建系统提示
        system_prompt = f"你是一个专业的{frag_type}经验生成专家，基于用户需求生成高质量的经验片段。"
        
        # 构建上下文
        context = f"任务: {task_description}\n"
        if dialog_context:
            context += f"对话上下文: {dialog_context}\n"
            
        # 根据片段类型构建不同的提示
        if frag_type == "WHY":
            prompt = f"""
{context}

请为以上任务生成一个完整的WHY经验片段，包含以下结构：
1. 目标 (goal)：明确的任务目标
2. 背景 (background)：任务的背景和原因
3. 约束条件 (constraints)：需要考虑的限制和条件，以数组形式提供
4. 预期效果 (expected_outcome)：完成任务后的预期结果

请以严格的JSON格式返回，格式如下：
{{
  "goal": "...",
  "background": "...",
  "constraints": ["约束1", "约束2", ...],
  "expected_outcome": "..."
}}

请确保返回的是可解析的标准JSON格式，不要添加额外的文本或说明。
"""
        elif frag_type == "HOW":
            prompt = f"""
{context}

请为以上任务生成一个完整的HOW经验片段，描述具体的实现步骤，每个步骤包含以下结构：
- page: 在哪个页面/环境进行操作
- action: 执行什么动作
- element: 操作的对象/元素
- intent: 该步骤的目的

请以严格的JSON格式返回，格式如下：
{{
  "steps": [
    {{
      "page": "...",
      "action": "...",
      "element": "...",
      "intent": "..."
    }},
    ...
  ]
}}

请生成3-5个详细步骤，确保返回的是可解析的标准JSON格式，不要添加额外的文本或说明。
"""
        elif frag_type == "CHECK":
            prompt = f"""
{context}

请为以上任务生成一个完整的CHECK经验片段，提供验证规则和检查点，格式如下：
{{
  "rules": [
    "验证规则1",
    "验证规则2",
    ...
  ]
}}

请生成3-5个关键验证规则，确保返回的是可解析的标准JSON格式，不要添加额外的文本或说明。
"""
        else:
            # 其他类型
            prompt = f"""
{context}

请为以上任务生成一个{frag_type}经验片段，包含相关内容。请以严格的JSON格式返回，确保可以被程序解析。
"""
            
        try:
            # 调用模型
            response = call_openai(prompt, system_prompt)
            logger.debug(f"模型原始响应: {response}")
            
            # 提取JSON
            json_str = extract_json_from_text(response)
            
            try:
                # 解析JSON
                fragment_data = json.loads(json_str)
                
                # 确保数据有正确的结构
                if frag_type == "WHY":
                    # 确保WHY有必要的字段
                    if not isinstance(fragment_data, dict):
                        fragment_data = {
                            "goal": task_description,
                            "background": "",
                            "constraints": [],
                            "expected_outcome": ""
                        }
                    elif "goal" not in fragment_data:
                        fragment_data["goal"] = task_description
                    if "constraints" not in fragment_data or not isinstance(fragment_data["constraints"], list):
                        fragment_data["constraints"] = []
                elif frag_type == "HOW":
                    # 确保HOW有steps字段
                    if not isinstance(fragment_data, dict) or "steps" not in fragment_data:
                        fragment_data = {"steps": []}
                    # 确保steps是列表
                    if not isinstance(fragment_data["steps"], list):
                        fragment_data["steps"] = []
                elif frag_type == "CHECK":
                    # 确保CHECK有rules字段
                    if not isinstance(fragment_data, dict) or "rules" not in fragment_data:
                        fragment_data = {"rules": []}
                    # 确保rules是列表
                    if not isinstance(fragment_data["rules"], list):
                        fragment_data["rules"] = []
                
                # 创建生成结果
                generated_result = [{
                    "fragment": {
                        "type": frag_type,
                        "data": fragment_data
                    },
                    "task_name": f"AI生成: {task_description}",
                    "similarity": 0.85,  # 给生成内容一个较高的初始相似度
                    "source": "ai_generated"  # 标记来源为AI生成
                }]
                
                return generated_result
                
            except json.JSONDecodeError as e:
                logger.error(f"解析生成的JSON失败: {str(e)}")
                return []
                
        except Exception as e:
            logger.error(f"生成{frag_type}片段失败: {str(e)}")
            return []
    
    def enhance_experience(self, experience_pack, dialog_context: str = None) -> Dict[str, Any]:
        """
        增强经验包质量
        
        Args:
            experience_pack: 经验包对象
            dialog_context: 对话上下文
            
        Returns:
            增强建议
        """
        # 记录交互
        if dialog_context:
            self.dialogue_history.append(f"用户: {dialog_context}")
            self.dialogue_history.append(f"系统: 为任务「{experience_pack.task_name}」提供增强建议")
            
        # 评估当前经验质量
        if not self.evaluator:
            return {
                "success": False,
                "message": "未配置评估器，无法进行质量评估"
            }
            
        evaluation = self.evaluator.evaluate_experience_pack(experience_pack)
        overall_score = evaluation.get("overall_score", 0)
        
        # 分析缺失和薄弱部分
        enhancement_suggestions = []
        missing_types = []
        weak_fragments = []
        
        # 检查片段类型完整性
        existing_types = set(frag.frag_type for frag in experience_pack.fragments)
        for key_type in ["WHY", "HOW", "CHECK"]:
            if key_type not in existing_types:
                missing_types.append(key_type)
                enhancement_suggestions.append(f"缺少{key_type}类型片段")
        
        # 检查片段质量
        for fragment in experience_pack.fragments:
            frag_result = self.evaluator.evaluate_fragment(fragment)
            frag_score = frag_result.get("score", 0)
            
            # 评分较低的片段需要增强
            if frag_score < 0.6:
                weak_fragments.append({
                    "type": fragment.frag_type,
                    "suggestions": frag_result.get("suggestions", [])
                })
                
                # 添加具体增强建议
                suggestions = frag_result.get("suggestions", [])
                if suggestions:
                    enhancement_suggestions.append(
                        f"{fragment.frag_type}片段需要改进: {', '.join(suggestions[:2])}"
                    )
        
        # 推荐互补片段
        complementary_recommendations = {}
        if missing_types:
            # 查找与当前任务相关的补充片段
            task_description = experience_pack.task_name
            
            # 如果有WHY片段，使用目标丰富查询
            for fragment in experience_pack.fragments:
                if fragment.frag_type == "WHY" and "goal" in fragment.data:
                    task_description += f" {fragment.data['goal']}"
                    break
            
            # 获取推荐（会自动补充生成）
            recommendations = self.recommend_for_task(task_description, experience_pack.fragments)
            complementary_recommendations = recommendations
        
        # 返回增强分析结果
        return {
            "success": True,
            "quality_level": "高" if overall_score >= 0.8 else "中" if overall_score >= 0.6 else "低",
            "missing_types": missing_types,
            "enhancement_suggestions": enhancement_suggestions,
            "complementary_recommendations": complementary_recommendations,
            "has_enhancement_potential": bool(enhancement_suggestions)
        }
    
    def learn_from_experience(self, experience_pack, dialog_context: str = None) -> Dict:
        """
        从经验中学习并保存
        
        Args:
            experience_pack: 经验包对象
            dialog_context: 对话上下文
            
        Returns:
            学习结果
        """
        # 处理对话上下文
        if dialog_context:
            dialogue_logs = [f"用户: {dialog_context}"]
            dialogue_logs.append(f"系统: 已记录关于「{experience_pack.task_name}」的经验")
            
            # 将对话添加到历史
            self.dialogue_history.extend(dialogue_logs)
            
            # 准备对话日志
            context = "\n".join(dialogue_logs)
        else:
            context = None
        
        # 添加到经验库
        result = self.retriever.add_experience(experience_pack, context)
        
        return {
            "success": result["success"],
            "message": f"已学习「{experience_pack.task_name}」相关经验"
        }
    
    def save_dialogue_history(self, exp_id: str = None) -> bool:
        """
        保存对话历史到经验
        
        Args:
            exp_id: 经验ID (如果不提供，则保存到新经验)
            
        Returns:
            是否保存成功
        """
        if not self.dialogue_history:
            return False
            
        # 如果提供了经验ID，尝试更新现有经验
        if exp_id is not None:
            for idx, exp in enumerate(self.retriever.experience_db):
                # 尝试匹配ID
                if exp.get("id") == exp_id:
                    # 更新对话日志
                    exp["dialogue_logs"] = self.dialogue_history
                    self.retriever.experience_db[idx] = exp
                    
                    # 保存更新
                    if self.retriever.db_path:
                        return self.retriever.save_db()
                    return True
        
        # 如果没有找到匹配经验或没有提供ID，创建新经验
        new_exp = {
            "task_title": "对话记录",
            "dialogue_logs": self.dialogue_history
        }
        
        # 添加到数据库
        self.retriever.experience_db.append(new_exp)
        
        # 重建索引
        self.retriever._build_indexes()
        
        # 保存更新后的数据库
        if self.retriever.db_path:
            return self.retriever.save_db()
        return True


# 示例用法
if __name__ == "__main__":
    # 配置路径
    db_path = "rich_expert_validation.json"
    
    # 初始化评估器和检索器
    evaluator = ExperienceEvaluator()
    retriever = ExperienceRetriever(db_path, evaluator)
    
    # 初始化推荐器
    recommender = FragmentRecommender(retriever, evaluator)
    
    # 测试推荐功能 - 为网站开发任务推荐经验片段
    recommendations = recommender.recommend_for_task(
        "开发一个自动化测试系统，用于验证网页界面变化",
        dialog_context="用户询问如何开发网页界面自动验证工具"
    )
    
    print("==== 推荐结果 ====")
    for frag_type, items in recommendations.items():
        print(f"\n{frag_type}类型推荐:")
        for item in items[:2]:  # 只显示前2个
            print(f"- 任务: {item['task_name']}")
            print(f"  相似度: {item['similarity']:.2f}")
            print(f"  来源: {'AI生成' if item.get('source') == 'ai_generated' else '经验库'}")
    
    # 测试学习功能
    from experienceagent.fragment_scorer import ExperienceEvaluator

    # 模拟经验片段和包
    class MockFragment:
        def __init__(self, frag_type, data):
            self.frag_type = frag_type
            self.data = data

    class MockExperiencePack:
        def __init__(self, task_name):
            self.task_name = task_name
            self.version = 1
            self.trust_score = 0.7
            self.fragments = []
            self.kg = None  # 假设有知识图属性

        def add_fragment(self, fragment):
            self.fragments.append(fragment)
            
    # 创建一个经验包
    exp_pack = MockExperiencePack("移动应用性能监控")
    exp_pack.add_fragment(MockFragment("WHY", {
        "goal": "实时监控移动应用性能",
        "background": "用户反馈应用偶尔卡顿",
        "constraints": ["低资源占用", "高精度"],
        "expected_outcome": "能够定位性能瓶颈"
    }))
    
    # 学习这个经验
    result = recommender.learn_from_experience(exp_pack, 
        dialog_context="用户询问如何监控移动应用性能，我建议了一种实时监控方案")
    
    print("\n==== 学习结果 ====")
    print(f"结果: {result['message']}")
    
    # 获取经验库统计
    stats = retriever.get_stats()
    print("\n==== 经验库统计 ====")
    print(f"总经验数: {stats['total']}")
    print("片段类型分布:")
    for ftype, count in stats["fragment_types"].items():
        print(f"- {ftype}: {count}")