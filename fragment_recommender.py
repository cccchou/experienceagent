#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
经验片段推荐系统
适配rich_expert_validation.json格式
添加GPT生成补充功能，当经验库没有相关内容时自动生成
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import os
import json
import logging
import time
import random
import re
from openai import OpenAI
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



# 片段类型到字段名的映射
TYPE_FIELD_MAP = {
    "WHY": "why_structured",
    "HOW": "how_behavior_logs",
    "CHECK": "check_rules",
    "DIALOGUE": "dialogue_logs"
}


class ExperienceRetriever:
    """
    经验检索器
    适配rich_expert_validation.json格式
    """
    
    def __init__(self, db_path: str = None, evaluator: ExperienceEvaluator = None):
        """
        初始化经验检索器
        
        Args:
            db_path: 经验数据库路径
            evaluator: 评分器实例
        """
        self.db_path = db_path
        self.evaluator = evaluator
        
        # 初始化数据库和索引
        self.experience_db = []
        self.type_index = defaultdict(list)  # 按片段类型索引
        self.tag_index = defaultdict(list)   # 按标签索引
        
        # 加载数据库
        if db_path and os.path.exists(db_path):
            self._load_db()
    
    def _load_db(self):
        """加载经验数据库"""
        try:
            logger.warning(f"加载经验数据库")
            with open(self.db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 可能是列表或字典格式
            if isinstance(data, dict):
                # 如果是字典，检查是否有数据字段
                self.experience_db = data.get('data', []) if any(key in data for key in ['data', 'experiences']) else [data]
            else:
                # 如果是列表，直接使用
                self.experience_db = data
                
            # 构建索引
            self._build_indexes()
            
            logger.info(f"已加载 {len(self.experience_db)} 条经验")
            return True
        except Exception as e:
            logger.error(f"加载经验数据库失败: {str(e)}")
            return False
    
    def _build_indexes(self):
        """构建索引"""
        self.type_index = defaultdict(list)
        self.tag_index = defaultdict(list)
        
        # 遍历经验库
        for idx, exp in enumerate(self.experience_db):
            # 片段类型索引
            for field_name, type_name in TYPE_FIELD_MAP.items():
                if type_name in exp:
                    self.type_index[type_name].append(idx)
            
            # 标签索引
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
        # 使用传入的路径或默认路径
        save_path = path or self.db_path
        if not save_path:
            logger.error("未指定保存路径")
            return False
            
        try:
            # 写入文件
            with open(save_path, 'w', encoding='utf-8') as f:
                # db_as_dicts = [frag.to_dict() for frag in self.experience_db]
                json.dump(self.experience_db, f, ensure_ascii=False, indent=2,default=lambda o: o.to_dict() if hasattr(o, "to_dict") 
                                  else o.__dict__)
            
            logger.info(f"已保存 {len(self.experience_db)} 条经验到 {save_path}")
            return True
        except Exception as e:
            logger.error(f"保存经验数据库失败: {str(e)}")
            return False
    
    def add_experience(self, experience_pack) -> Dict:
        """
        添加经验到数据库，适配rich_expert_validation.json格式
        
        Args:
            experience_pack: 经验包对象
            
            
        Returns:
            添加结果
        """
        # 转换为rich_expert_validation格式
        exp_data = {
        }
        
        # 转换片段为对应格式
        for fragment in experience_pack.fragments:
            logger.info(f"添加片段类型: {fragment.frag_type}")
            logger.info(f"片段内容: {json.dumps(fragment.data, ensure_ascii=False)[:100]}...")
            
            if fragment.frag_type == "WHY":
                exp_data["why_structured"] = fragment.data
            elif fragment.frag_type == "HOW":
                behavior_logs = []
                if 'steps' in fragment.data:
                    fragment = fragment.data['steps']
                # 转换HOW片段为how_behavior_logs格式
                    for steps in fragment:
                        if isinstance(steps, dict):
                            behavior_log = {
                                "page": steps.get("page", "未指定页面"),
                                "action": steps.get("action", "执行"),
                                "element": steps.get("element", "未指定元素"),
                                "intent": steps.get("description", steps.get("intent", "未指定意图"))
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
        
        logger.info(f"已添加新经验: {experience_pack.task_name}")
        
        return {
            "success": True,
            "index": len(self.experience_db) - 1,
        }
    
    def get_by_fragment_type(self, frag_type: str) -> List[Dict]:
        """
        按片段类型检索经验
        
        Args:
            frag_type: 片段类型 (WHY, HOW, CHECK)
            
        Returns:
            匹配经验列表
        """
        field_mapping = TYPE_FIELD_MAP.get(frag_type)
        if not field_mapping or field_mapping not in self.type_index:
            return []
        
        results = []
        for idx in self.type_index[field_mapping]:
            exp = self.experience_db[idx]
            
            # 获取对应的片段字段
            field_name = TYPE_FIELD_MAP.get(frag_type)
            if field_name in exp:
                result = {
                    "fragment": exp[field_name],
                    "task_name": exp.get('task_title', '未命名任务')
                }
                results.append(result)
        
        return results
    
    def semantic_search(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """
        语义搜索经验
        
        Args:
            query_text: 查询文本,为一个任务的task描述
            top_k: 返回结果数量
            
        Returns:
            相关经验列表:list[Dict]
            {"experience": match["experience"],
            "similarity": match["similarity"],
            "reason": match.get("reason", "")}
        """
        try:
            # 准备候选项
            candidates = []
            for idx, exp in enumerate(self.experience_db):
                text_repr = self._get_experience_task(exp)
                candidates.append({
                    "id": idx,
                    "text": text_repr,
                    "experience": exp
                })
                
            # 语义匹配
            matches = self._semantic_matching(query_text, candidates, top_k)
            
            # 整理结果
            result = []
            for match in matches:
                result.append({
                    "experience": match["experience"],
                    "similarity": match["similarity"],
                    "reason": match.get("reason", "")
                })
                
            return result
        except Exception as e:
            logger.error(f"语义搜索失败: {str(e)}")
            return self._fallback_text_search(query_text, top_k)
    
    def _semantic_matching(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        """
        语义匹配
        
        Args:
            query: 查询文本
            candidates: 候选项列表:list[Dict]
            {"id": idx,
            "text": text_repr,
            "experience": exp}
            top_k: 返回数量
            
        Returns:
            matches:list[Dict]:
            {"experience": candidates[candidate_id]["experience"],
            "similarity": similarity,
            "reason": item.get("reason", "相关内容匹配")}
        """
        # 准备候选项文本
        candidate_texts = []
        for candidate in candidates:
            candidate_texts.append(candidate["text"])
        logger.info(f'候选集任务为: {candidate_texts}')
        
        # 构建提示词
        prompt = f"""请作为专业的经验检索专家，从候选项中找出与查询最相关的{top_k}个经验:

        查询: {query}

        候选经验:
        """

        for i, text in enumerate(candidate_texts):  # 是否限制候选项数量需要后期迭代提升考虑
            prompt += f"{i+1}. {text}\n\n"
        
        prompt += f"""

            请返回严格的JSON格式结果，不要添加任何额外说明或标记，只返回一个包含以下结构的JSON数组:
            [
            {{"id": 经验ID, "similarity": 相关性评分, "reason": "匹配原因简述"}},
            ...
            ]

            相关性评分必须是0-1之间的数字。按相关性从高到低排序，只返回最相关的{top_k}个结果。
            **注意：**
            - **只输出纯 JSON**，不要使用任何 ```、```json 或其他代码块标记；
            - 不要添加额外文字、注释或解释；
            - 确保输出可以被 `json.loads()` 直接解析。
            """
        
        try:
            # 调用模型
            response = call_openai(prompt)
            
            # 解析结果
            # logger.info(f"原始模型响应: {response}")
            try:
                results = json.loads(response)
                logger.info(f"原始模型解码后: {response}")
            except json.JSONDecodeError as e:
                raise ValueError(f"模型返回的JSON格式错误: {str(e)}\n响应内容: {response}")
            
            if not isinstance(results, list):
                logger.error(f"模型返回的不是一个列表，而是: {type(results)}")
                return []
                
            # 整理结果
            matches = []
            for item in results:
                candidate_id = int(item.get("id", 0)) - 1
                if 0 <= candidate_id < len(candidates):
                    # 确保相似度在有效范围内
                    similarity = item.get("similarity", 0.0)
                    similarity = min(1.0, max(0.0, float(similarity)))
                    
                    matches.append({
                        "experience": candidates[candidate_id]["experience"],
                        "similarity": similarity,
                        "reason": item.get("reason", "相关内容匹配")
                    })
            
            # 按相似度排序
            matches.sort(key=lambda x: x["similarity"], reverse=True)
            # logger.info(f"匹配结果: {matches}")
            # 限制返回数量
            return matches[:top_k]
        except Exception as e:
            raise ValueError(f"语义匹配失败: {str(e)}\n查询: {query}\n候选项: {candidates}") from e
    
    
    # def _fallback_text_search(self, query: str, candidates = None, top_k: int = 5) -> List[Dict]:
    #     """简单文本匹配的备用搜索"""
    #     # 准备查询词
    #     query_terms = set(term.lower() for term in query.split() if len(term) > 3)
        
    #     matches = []
        
    #     # 如果提供了候选项，使用它们
    #     if isinstance(candidates, list) and all(isinstance(c, dict) for c in candidates):
    #         for candidate in candidates:
    #             # 计算匹配词数
    #             text = candidate["text"].lower()
    #             match_count = sum(1 for term in query_terms if term in text)
                
    #             # 计算相似度（简单匹配比例）
    #             similarity = match_count / len(query_terms) if query_terms else 0.1
                
    #             if similarity > 0:
    #                 matches.append({
    #                     "experience": candidate["experience"],
    #                     "similarity": min(similarity, 0.9),  # 上限0.9，不要太自信
    #                     "reason": "文本关键词匹配"
    #                 })
    #     else:
    #         # 如果没有提供候选项，使用整个数据库
    #         for idx, exp in enumerate(self.experience_db):
    #             text = self._get_experience_text(exp).lower()
    #             match_count = sum(1 for term in query_terms if term in text)
                
    #             # 计算相似度（简单匹配比例）
    #             similarity = match_count / len(query_terms) if query_terms else 0.1
                
    #             if similarity > 0:
    #                 matches.append({
    #                     "experience": exp,
    #                     "similarity": min(similarity, 0.9),  # 上限0.9，不要太自信
    #                     "reason": "文本关键词匹配"
    #                 })
        
    #     # 按相似度排序
    #     matches.sort(key=lambda x: x["similarity"], reverse=True)
        
    #     # 限制返回数量
    #     return matches[:top_k]
    
    def _get_experience_task(self, experience: Dict) -> str:
        """获取经验的文本的expected_outcome标签，根据expected_outcome去和query进行匹配"""
        parts = ''
        # WHY部分
        why_data = experience.get("why_structured", {})
        if why_data:
            # parts.append("--- WHY ---")
            if "goal" in why_data:
                parts = why_data['goal']
        else:
            raise ValueError("经验中缺少WHY部分,匹配为空")
            # if "background" in why_data:
            #     parts.append(f"背景: {why_data['background']}")
            # if "constraints" in why_data:
            #     constraints = why_data["constraints"]
            #     if isinstance(constraints, list):
            #         parts.append(f"约束: {', '.join(constraints)}")
            # if "expected_outcome" in why_data:
            #     parts.append(f"预期效果: {why_data['expected_outcome']}")
        
        # # HOW部分
        # if "how_behavior_logs" in experience:
        #     parts.append("\n--- HOW ---")
        #     logs = experience["how_behavior_logs"]
        #     parts.append(f"步骤数量: {len(logs)}")
            
        #     for i, log in enumerate(logs[:3]):  # 只取前3步以简化
        #         if isinstance(log, dict):
        #             step_text = f"{i+1}: "
        #             if "action" in log:
        #                 step_text += f"{log['action']} "
        #             if "element" in log:
        #                 step_text += f"{log['element']} "
        #             if "page" in log:
        #                 step_text += f"在 {log['page']} 页面"
        #             if "intent" in log:
        #                 step_text += f"，目的是 {log['intent']}"
        #             parts.append(step_text)
        
        # # CHECK部分
        # if "check_rules" in experience:
        #     parts.append("\n--- CHECK ---")
        #     rules = experience["check_rules"]
        #     parts.append(f"规则数量: {len(rules)}")
            
        #     for i, rule in enumerate(rules[:3]):  # 只取前3条规则
        #         parts.append(f"{i+1}: {rule}")
        
        # # 对话日志
        # if "dialogue_logs" in experience:
        #     parts.append("\n--- 对话日志 ---")
        #     logs = experience["dialogue_logs"]
        #     for log in logs[:5]:  # 只取前5条对话
        #         parts.append(log)
        
        # # 片段信息
        # if "fragments" in experience:
        #     parts.append("\n--- 经验片段 ---")
        #     for fragment in experience.get("fragments", []):
        #         frag_type = fragment.get("type", "")
        #         frag_data = fragment.get("data", {})
        #         parts.append(f"\n--- {frag_type} ---")
        #         parts.append(json.dumps(frag_data, ensure_ascii=False)[:100])  # 限制长度
        
        # # 标签
        # if "tags" in experience:
        #     parts.append(f"\n标签: {', '.join(experience['tags'])}")
            
        return parts
    
    def recommend_complementary(self, fragments: List, top_k: int = 2) -> Dict[str, List]:
        """
        推荐互补的经验片段
        
        Args:
            fragments: 已有片段列表
            top_k: 每种类型返回的推荐数量
            
        Returns:
            按类型分类的推荐片段
        """
        # 确定已有的片段类型
        existing_types = set(fragment.frag_type for fragment in fragments if hasattr(fragment, "frag_type"))
        
        # 针对每种类型推荐
        recommendations = {}
        for frag_type in ["WHY", "HOW", "CHECK"]:
            if frag_type in existing_types:
                continue
                
            # 获取该类型的推荐
            fragments = self.get_by_fragment_type(frag_type)
            if fragments:
                recommendations[frag_type] = fragments[:top_k]
        
        return recommendations


class FragmentRecommender:
    """
    经验片段推荐器
    基于经验库提供智能推荐功能，并记录客户交互
    增加AI生成推荐功能
    """
    
    def __init__(self, retriever: ExperienceRetriever = None, evaluator: ExperienceEvaluator = None, 
                 db_path: str = None, min_recommendations: int = 1, min_similarity: float = 0.9):
        """
        初始化推荐器
        
        Args:
            retriever: 经验检索器
            evaluator: 评分器
            db_path: 数据库路径
            min_recommendations: 最少推荐数量
            min_similarity: 最低相似度阈值
        """
        if not retriever and not db_path:
            raise ValueError("必须提供检索器或数据库路径")
            
        self.retriever = retriever or ExperienceRetriever(db_path, evaluator)
        self.evaluator = evaluator
        self.dialogue_history = []
        self.min_recommendations = min_recommendations
        self.min_similarity = min_similarity
    
    def recommend_for_task(self, task_description: str, 
                          dialog_context: list = None) -> Dict[str, List]:
        """
        基于任务描述推荐经验片段
        如果经验库中没有足够相关内容，自动生成推荐
        
        Args:
            task_description: 任务描述
            dialog_context: 对话上下文
            
        Returns:
            推荐结果
        """
        # 记录交互
        system_dialogs=['请问你的目标是什么?','你为什么需要这个功能?','有哪些限制条件我们要考虑?','你希望最终达到什么样的效果?']
        if len(dialog_context)>0:
            for i,c in enumerate(dialog_context):
                self.dialogue_history.append(f"用户: {c}")
                self.dialogue_history.append(f'系统：{system_dialogs[i]}')
        
        # 查找相似经验
        similar_experiences = self.retriever.semantic_search(
            task_description, top_k=1
        )
        # logger.info(f'相似经验为{similar_experiences}')
        # 提取全面推荐 - 按片段类型组织
        recommendations = {}
        recommendations["WHY"] = []
        recommendations["HOW"] = []
        recommendations["CHECK"] = []
        fragment_sources = {}  # 追踪片段来源
        fragment_sources["WHY"] = set()
        fragment_sources["HOW"] = set()
        fragment_sources["CHECK"] = set()
        cnt=0
        for item in similar_experiences:
            exp = item["experience"]
            similarity = item["similarity"]
            reason = item['reason']
            
            # 跳过低相似度的结果
            if similarity < self.min_similarity:
                continue
                
            # 获取任务名称
            task_name = exp.get("task_title", "")
            if not task_name:
                task_name = task_description
                
            # 处理rich_expert_validation.json格式
            # 检查WHY类型
            if "why_structured" in exp :
                recommendations["WHY"].append({
                        "fragment": {
                            "type": "WHY",
                            "data": exp["why_structured"]
                        },
                        "task_name": task_name,
                        "similarity": similarity,
                        'reason': reason,
                        "source": "database"  # 标记来源为数据库
                    })
                fragment_sources["WHY"].add(task_name)
                    
            # 检查HOW类型
            if "how_behavior_logs" in exp:
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
            if "check_rules" in exp:
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
            cnt+=1
                    
            # # 兼容fragments格式
            # if "fragments" in exp:
            #     for fragment in exp["fragments"]:
            #         frag_type = fragment.get("type")
                    
            #         # 跳过已有类型
            #         if frag_type in existing_types:
            #             continue
                        
            #         # 创建类型条目
            #         if frag_type not in recommendations:
            #             recommendations[frag_type] = []
            #             fragment_sources[frag_type] = set()
                        
            #         # 避免重复推荐相同经验中的片段
            #         if task_name not in fragment_sources[frag_type]:
            #             recommendations[frag_type].append({
            #                 "fragment": fragment,
            #                 "task_name": task_name,
            #                 "similarity": similarity,
            #                 "source": "database"  # 标记来源为数据库
            #             })
            #             fragment_sources[frag_type].add(task_name)
        
        # 对每种类型的推荐结果排序
        for frag_type in recommendations:
            recommendations[frag_type].sort(
                key=lambda x: x["similarity"],
                reverse=True
            )
        logger.info(f"经验库的推荐结果: {json.dumps(recommendations, ensure_ascii=False, indent=2)}")
            
        # 检查是否需要GPT生成补充推荐
        generated_contents = {} # 跟踪生成的内容
        
        if cnt == 0:
            logger.info(f"没有找到与任务「{task_description}」相关的经验，将尝试AI生成推荐片段")        
            # 如果某种类型的推荐不足，使用AI生成补充
            for frag_type in ["WHY", "HOW", "CHECK"]:
                generated_fragments = self._generate_fragment(task_description, frag_type)
                
                if generated_fragments:
                    # 确保推荐列表已初始化
                    if frag_type not in recommendations:
                        recommendations[frag_type] = []
                    
                    # 添加生成的片段到本地变量
                    generated_contents[frag_type] = []
                    
                    # 添加生成的片段
                    for gen_fragment in generated_fragments:
                        recommendations[frag_type].append(gen_fragment)
                        generated_contents[frag_type].append(gen_fragment)
                    
                    # 重新排序
                    recommendations[frag_type].sort(
                        key=lambda x: x["similarity"],
                        reverse=True
                    )
                    
        # 保存生成的内容到经验库 - 与推荐逻辑分开处理
        if generated_contents:
            logger.info(f"正在保存AI生成内容到经验库...")
            try:
                from experienceagent.controller_agent import ExperiencePack, ExperienceFragment
                
                # 为每种生成的类型创建单独的经验包
                # 创建一个经验包
                pack_name = f"AI生成: {task_description}"
                exp_pack = ExperiencePack(pack_name)
                for frag_type, fragments in generated_contents.items():
                    for gen_fragment in fragments:
                        # 创建一个经验
                        # 添加片段
                        frag_data = gen_fragment["fragment"]["data"]
                        # 打印详细日志以便调试
                        logger.info(f"添加到经验库的{frag_type}片段数据: {json.dumps(frag_data, ensure_ascii=False)}")
                        
                        # 创建片段实例
                        exp_fragment = ExperienceFragment(frag_type, frag_data)
                        exp_pack.add_fragment(exp_fragment)
                        
                        # # 添加到经验库
                        # context = f"AI为查询「{task_description}」生成的{frag_type}片段"
                        # if dialog_context:
                        #     dialog_context = f"{dialog_context} -> {context}"
                    # 强制同步到文件
                result = self.retriever.add_experience(exp_pack)
                self.retriever.save_db()  # 显式保存到文件
                
                if result.get("success", False):
                    logger.info(f"已将AI生成的{frag_type}片段添加到经验库，ID: {result.get('index', '')}")
                else:
                    logger.warning(f"保存AI生成的{frag_type}片段失败")
            except Exception as e:
                logger.error(f"保存AI生成内容到经验库时出错: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                
        return recommendations
    


    def _generate_fragment(self, task_description: str, frag_type: str) -> List[Dict]:
        """
        使用AI生成特定类型的片段
        
        Args:
            task_description: 任务描述
            frag_type: 片段类型
            
        Returns:
            生成的片段列表
        """
        logger.info(f"为任务「{task_description}」生成{frag_type}片段")
        
        # 构建系统提示
        system_prompt = f"你是一个专业的经验生成专家。你的任务是按照要求的格式生成JSON，确保结构完全符合规范，没有任何多余文本。"
        
        # 构建上下文
        context = f"任务: {task_description}\n"
            
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

                **注意：**
            - **只输出纯 JSON**，不要使用任何 ```、```json 或其他代码块标记；
            - 不要添加额外文字、注释或解释；
            - 确保输出可以被 `json.loads()` 直接解析。
            """
        elif frag_type == "HOW":
            prompt = f"""
            {context}

            请为以上任务生成一个HOW经验片段，描述具体的实现步骤。

            你需要返回一个具有以下结构的JSON对象，包含多个步骤：
            {{
            "steps": [
                {{
                "page": "步骤1所在页面",
                "action": "步骤1要执行的动作",
                "element": "步骤1操作的元素",
                "intent": "步骤1的目的"
                }},
                {{
                "page": "步骤2所在页面",
                "action": "步骤2要执行的动作",
                "element": "步骤2操作的元素", 
                "intent": "步骤2的目的"
                }},
                ...其他步骤
            ]
            }}

            每个步骤必须包含这四个字段：
            - page: 在哪个页面/环境进行操作
            - action: 执行什么动作（如点击、输入、选择等）
            - element: 操作的对象/元素（如按钮、输入框、下拉菜单等）
            - intent: 该步骤的目的

            **注意：**
            - **只输出纯 JSON**，不要使用任何 ```、```json 或其他代码块标记；
            - 不要添加额外文字、注释或解释；
            - 确保输出可以被 `json.loads()` 直接解析。
            """
        elif frag_type == "CHECK":
            prompt = f"""
            {context}

            请为以上任务生成一个完整的CHECK经验片段，提供3-5个具体的验证规则。

            你需要返回一个具有以下结构的JSON对象：
            {{
            "rules": [
                "具体的验证规则1",
                "具体的验证规则2",
                "具体的验证规则3",
                ...更多规则
            ]
            }}

            规则应该是具体、可操作的验证标准，例如：
            - "每个页面元素必须具备唯一定位属性"
            - "页面加载时间不应超过3秒"
            - "所有表单提交必须有成功/失败的反馈信息"

                **注意：**
            - **只输出纯 JSON**，不要使用任何 ```、```json 或其他代码块标记；
            - 不要添加额外文字、注释或解释；
            - 确保输出可以被 `json.loads()` 直接解析。
            """
        else:
            # 其他类型
            prompt = f"""
            {context}

            请为以上任务生成一个{frag_type}经验片段，包含相关内容。请以严格的JSON格式返回，确保可以被程序解析。
            **注意：**
            - **只输出纯 JSON**，不要使用任何 ```、```json 或其他代码块标记；
            - 不要添加额外文字、注释或解释；
            - 确保输出可以被 `json.loads()` 直接解析。
            """
            
        try:
            # 调用模型
            response = call_openai(prompt, system_prompt)
            logger.info(f"模型原始响应: {response}")
            try:
                # 解析JSON
                fragment_data = json.loads(response)
                logger.info(f"解析后的数据结构: {type(fragment_data)}")
                logger.info(f"解析后的数据内容: {fragment_data}")
                # 确保数据有正确的结构
                if frag_type == "WHY":
                    # 确保WHY有必要的字段
                    if not isinstance(fragment_data, dict):
                        logger.warning("WHY数据不是字典格式，使用默认结构")
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
                    if not isinstance(fragment_data, dict):
                        logger.warning("HOW数据不是字典格式，使用默认结构")
                        fragment_data = {"steps": []}
                    elif "steps" not in fragment_data or not isinstance(fragment_data["steps"], list):
                        fragment_data["steps"] = []
                    # 确保每个步骤包含所有必要字段
                    for step in fragment_data['steps']:
                    # 确保HOW有steps字段
                        if not isinstance(step, dict):
                            logger.warning("HOW数据不是字典格式，使用默认结构")
                            fragment_data = {"steps": []}
                        else:
                            # 确保每个步骤包含所有必要字段
                            for field in ["page", "action", "element", "intent"]:
                                if field not in step or not step[field]:
                                    step[field] = "未指定" + field
                elif frag_type == "CHECK":
                    # 确保CHECK有rules字段
                    # 确保rules是列表格式
                    if not isinstance(fragment_data, dict):
                        logger.warning("CHECK数据不是字典格式，使用默认结构")
                        fragment_data = {"rules": []}
                    elif "rules" not in fragment_data or not isinstance(fragment_data["rules"], list):
                        fragment_data["rules"] = []
                    if len(fragment_data) == 0:
                        # 如果规则为空，添加默认规则
                        fragment_data = [
                            f"确保{task_description}功能正确实现",
                            "验证所有用户输入被正确处理",
                            "测试界面元素正确响应用户交互"
                        ]
                
                # 创建生成结果
                generated_result = [{
                    "fragment": {
                        "type": frag_type,
                        "data": fragment_data
                    },
                    "task_name": f"AI生成: {task_description}",
                    "similarity": 1,  # 给生成内容一个较高的初始相似度
                    "source": "ai_generated",  # 标记来源为AI生成
                    "reason": "AI生成的推荐片段"
                }]
                
                # 再次验证数据完整性
                if frag_type == "HOW":
                    steps = fragment_data.get("steps", [])
                    if steps:
                        logger.info(f"生成了{len(steps)}个HOW步骤")
                        for i, step in enumerate(steps):
                            logger.info(f"步骤{i+1}: {json.dumps(step, ensure_ascii=False)}")
                    else:
                        logger.warning("生成的HOW步骤为空")
                elif frag_type == "CHECK":
                    rules = fragment_data.get("rules", [])
                    if rules:
                        logger.info(f"生成了{len(rules)}条CHECK规则")
                        for i, rule in enumerate(rules):
                            logger.info(f"规则{i+1}: {rule}")
                    else:
                        logger.warning("生成的CHECK规则为空")
                
                # 记录生成结果
                logger.info(f"成功生成{frag_type}片段，准备添加到经验库")
                
                return generated_result
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON解析失败: {str(e)}")
                logger.error("生成的内容可能不符合预期格式，请检查模型响应")
                return []
                    
        except Exception as e:
            logger.error(f"生成{frag_type}片段失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    #todo
    def enhance_experience(self, experience_pack) -> Dict:
        """
        增强经验包质量
        
        Args:
            experience_pack: 经验包对象
            
        Returns:
            增强建议
        """
        
        # 评估经验包质量
        if not self.evaluator:
            return {
                "success": False,
                "message": "未配置评估器，无法进行质量评估",
                "has_enhancement_potential": False
            }
        
        # 获取当前评分
        evaluation = self.evaluator.evaluate_experience(experience_pack)
        overall_score = evaluation["overall_score"]
        
        # 根据评分判断质量级别
        if overall_score >= 0.8:
            quality_level = "高"
        elif overall_score >= 0.5:
            quality_level = "中"
        else:
            quality_level = "低"
            
        # 查找弱项
        weak_fragments = []
        for frag_type, frag_result in evaluation["fragment_scores"].items():
            frag_score = frag_result["score"]
            if frag_score < 0.7:
                weak_fragments.append(frag_type)
        
        # 识别缺失的片段类型
        existing_types = set(fragment.frag_type for fragment in experience_pack.fragments)
        missing_types = set(["WHY", "HOW", "CHECK"]) - existing_types
        
        # 准备改进建议
        suggestions = []
        
        # 添加关于弱项的建议
        for weak_type in weak_fragments:
            key_type = weak_type
            if key_type in evaluation["fragment_scores"]:
                for issue in evaluation["fragment_scores"][key_type].get("issues", []):
                    suggestions.append(f"{key_type}片段的{issue}")
        
        # 添加关于缺失类型的建议
        for missing_type in missing_types:
            suggestions.append(f"缺少{missing_type}类型片段")
        
        # 获取互补推荐
        complementary_recommendations = {}
        if missing_types:
            # 使用任务名称获取推荐
            task_description = experience_pack.task_name
            
            # 记录需要增强的类型
            for type_to_enhance in missing_types:
                # 生成或获取推荐
                fragments = self._generate_fragment(task_description, type_to_enhance)
                
                if fragments:
                    complementary_recommendations[type_to_enhance] = fragments
        
        return {
            "success": True,
            "quality_level": quality_level,
            "overall_score": overall_score,
            "weak_fragments": weak_fragments,
            "missing_types": list(missing_types),
            "has_enhancement_potential": bool(weak_fragments or missing_types),
            "suggestions": suggestions,
            "complementary_recommendations": complementary_recommendations
        }
    
    def save_dialogue_history(self) -> bool:
        """
        保存对话历史到经验
        
        Args:
            exp_id: 经验ID (如果不提供，则保存到新经验)
            
        Returns:
            是否保存成功
        """
        if not self.dialogue_history:
            return False
            
        try:
            from experienceagent.controller_agent import ExperiencePack
            
            # 检查是否已有经验
            
            exp = self.retriever.experience_db[-1]
            if "dialogue_logs" in exp:
                exp["dialogue_logs"].extend(self.dialogue_history)
            else:
                exp["dialogue_logs"] = self.dialogue_history
            
            # 保存更改
            self.retriever.save_db()
            return True
        except Exception as e:
            logger.error(f"保存对话历史失败: {str(e)}")
            return False


if __name__ == "__main__":
    # 示例
    db_path = "rich_expert_validation.json"
    evaluator = ExperienceEvaluator()
    retriever = ExperienceRetriever(db_path, evaluator)
    recommender = FragmentRecommender(retriever, evaluator)
    
    # 测试推荐
    task = "创建一个能够实现双12广告流量数据分析功能网站，响应速度快、平台来源多的网站"
    context = ["用户询问如何开发网页界面自动验证",'a','b','c']
    
    result = recommender.recommend_for_task(task, context)
    
    # 显示结果
    print(f"=== 推荐 ===")
    items = result['WHY']
    for item in items:
        print(f"- 任务: {item['fragment']['data']['goal']}")
        print(f"  相似度: {item['similarity']:.2f}")
        print(f"  来源: {'AI生成' if item.get('source') == 'ai_generated' else '经验库'}")
        print(f'  原因: {item.get("reason", "无")}')
    
    # 保存对话
    # recommender.save_dialogue_history()
    

    
    # 显示结果
    print("\n==== 学习结果 ====")
    print(f"结果: {result}")
    
    # 显示经验库统计
    print("\n==== 经验库统计 ====")
    print(f"总经验数: {len(retriever.experience_db)}")
    print("片段类型分布:")
    for type_name, indices in retriever.type_index.items():
        print(f"- {type_name}: {len(indices)}")