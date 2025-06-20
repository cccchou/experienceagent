# fragment_recommender.py

from typing import List, Dict, Any
from openai import OpenAI
import json

client = OpenAI(
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



class ExperienceRetriever:
    """
    经验检索器
    用于从经验库中查找和检索相关经验
    """
    def __init__(self, db_path: str = None):
        """
        初始化检索器
        
        Args:
            db_path: 经验库文件路径（可选）
        """
        self.db_path = db_path
        self.experience_db = []
        
        # 如果提供了路径，尝试加载经验库
        if db_path and os.path.exists(db_path):
            self._load_db()
    
    def _load_db(self):
        """从文件加载经验库"""
        try:
            with open(self.db_path, 'r', encoding='utf-8') as f:
                self.experience_db = json.load(f)
            return True
        except Exception as e:
            print(f"加载经验库失败: {str(e)}")
            self.experience_db = []
            return False
    
    def save_db(self, path: str = None):
        """
        保存经验库到文件
        
        Args:
            path: 保存路径，如不提供则使用初始化时的路径
            
        Returns:
            是否保存成功
        """
        save_path = path or self.db_path
        if not save_path:
            return False
            
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(self.experience_db, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存经验库失败: {str(e)}")
            return False
    
    def add_experience(self, experience_pack):
        """
        添加经验包到数据库
        
        Args:
            experience_pack: 经验包对象
        """
        # 转换经验包为可存储格式
        exp_dict = {
            "task_name": experience_pack.task_name,
            "version": experience_pack.version,
            "trust_score": experience_pack.trust_score,
            "fragments": []
        }
        
        # 添加所有片段
        for fragment in experience_pack.fragments:
            exp_dict["fragments"].append({
                "type": fragment.frag_type,
                "data": fragment.data
            })
        
        self.experience_db.append(exp_dict)
    
    def retrieve_by_task_name(self, task_name: str, threshold: float = 0.5) -> List[Dict]:
        """
        根据任务名称检索经验
        
        Args:
            task_name: 任务名称
            threshold: 相似度阈值
            
        Returns:
            匹配的经验列表
        """
        results = []
        task_name_lower = task_name.lower()
        
        for exp in self.experience_db:
            exp_task = exp.get("task_name", "").lower()
            # 简单相似度：包含关系或编辑距离
            if task_name_lower in exp_task or exp_task in task_name_lower:
                similarity = 0.8  # 高相似度
            else:
                # 计算词重叠度
                words1 = set(task_name_lower.split())
                words2 = set(exp_task.split())
                if not words1 or not words2:
                    similarity = 0.0
                else:
                    intersection = len(words1.intersection(words2))
                    union = len(words1.union(words2))
                    similarity = intersection / union
            
            if similarity >= threshold:
                results.append({
                    "experience": exp,
                    "similarity": similarity
                })
        
        # 按相似度降序排序
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results
    
    def retrieve_by_goal(self, goal: str, top_k: int = 3) -> List[Dict]:
        """
        根据目标描述检索经验
        使用语义匹配
        
        Args:
            goal: 目标描述
            top_k: 返回结果数量
            
        Returns:
            相似经验列表
        """
        if not self.experience_db or not goal:
            return []
            
        # 准备候选经验列表
        candidates = []
        for idx, exp in enumerate(self.experience_db):
            # 从WHY片段中提取目标
            exp_goal = ""
            for frag in exp.get("fragments", []):
                if frag.get("type") == "WHY" and "data" in frag:
                    frag_data = frag.get("data", {})
                    if "goal" in frag_data:
                        exp_goal = frag_data["goal"]
                        break
                        
            if exp_goal:
                candidates.append({
                    "id": idx,
                    "task": exp.get("task_name", ""),
                    "goal": exp_goal
                })
        
        if not candidates:
            return []
            
        # 使用OpenAI进行语义匹配
        prompt = f"""
请比较以下目标描述，找出与查询目标最相似的{top_k}个:

查询目标: {goal}

候选目标:
{json.dumps(candidates, ensure_ascii=False)}

请返回最相似的{top_k}个目标的ID，格式如下:
[
  {{"id": 候选ID, "similarity": 0.1-1.0之间的相似度, "reason": "简短相似原因"}}
]
"""
        
        try:
            response = call_openai(prompt)
            matches = json.loads(response)
            
            # 获取对应的经验
            results = []
            for match in matches:
                idx = match.get("id")
                if isinstance(idx, int) and 0 <= idx < len(self.experience_db):
                    exp = self.experience_db[idx]
                    results.append({
                        "experience": exp,
                        "similarity": match.get("similarity", 0),
                        "reason": match.get("reason", "")
                    })
                    
            return results
        
        except Exception as e:
            print(f"语义匹配失败: {str(e)}")
            # 备用方案：简单关键词匹配
            goal_words = set(goal.lower().split())
            results = []
            
            for idx, candidate in enumerate(candidates):
                cand_goal = candidate["goal"].lower()
                cand_words = set(cand_goal.split())
                
                if not goal_words or not cand_words:
                    continue
                    
                # 计算词重叠度
                intersection = len(goal_words.intersection(cand_words))
                union = len(goal_words.union(cand_words))
                similarity = intersection / union if union > 0 else 0
                
                if similarity > 0.3:  # 基础阈值
                    results.append({
                        "experience": self.experience_db[candidate["id"]],
                        "similarity": similarity,
                        "reason": "关键词匹配"
                    })
            
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:top_k]
    
    def retrieve_by_fragment_type(self, frag_type: str) -> List[Dict]:
        """
        根据片段类型检索
        
        Args:
            frag_type: 片段类型，如"WHY"、"HOW"、"CHECK"
            
        Returns:
            包含该类型片段的经验列表
        """
        results = []
        
        for exp in self.experience_db:
            fragments = exp.get("fragments", [])
            
            for fragment in fragments:
                if fragment.get("type") == frag_type:
                    results.append({
                        "task_name": exp.get("task_name", ""),
                        "version": exp.get("version", 1),
                        "trust_score": exp.get("trust_score", 0.5),
                        "fragment": fragment
                    })
                    break  # 每个经验只返回一次
                    
        return results
    
    def retrieve_similar_fragments(self, query_text: str, frag_type: str = None, top_k: int = 3) -> List[Dict]:
        """
        检索与查询文本相似的片段
        
        Args:
            query_text: 查询文本
            frag_type: 片段类型筛选（可选）
            top_k: 返回结果数量
            
        Returns:
            相似片段列表
        """
        # 收集所有符合类型的片段
        candidates = []
        
        for exp_idx, exp in enumerate(self.experience_db):
            for frag_idx, frag in enumerate(exp.get("fragments", [])):
                # 类型筛选
                if frag_type and frag.get("type") != frag_type:
                    continue
                    
                candidates.append({
                    "exp_idx": exp_idx,
                    "frag_idx": frag_idx,
                    "task": exp.get("task_name", ""),
                    "type": frag.get("type", ""),
                    "data": frag.get("data", {})
                })
        
        if not candidates:
            return []
            
        # 简单实现：比较内容相似度
        results = []
        query_text_lower = query_text.lower()
        
        for candidate in candidates:
            # 将片段数据转为文本
            frag_text = json.dumps(candidate["data"], ensure_ascii=False).lower()
            
            # 计算简单相似度
            words1 = set(query_text_lower.split())
            words2 = set(frag_text.split())
            if not words1 or not words2:
                similarity = 0.0
            else:
                # 计算重叠词的比例
                matches = 0
                for word in words1:
                    if word in frag_text and len(word) > 3:  # 只计算有意义的词
                        matches += 1
                similarity = matches / len(words1) if words1 else 0
            
            if similarity > 0.2:  # 基础阈值
                exp_idx = candidate["exp_idx"]
                frag_idx = candidate["frag_idx"]
                
                if 0 <= exp_idx < len(self.experience_db):
                    exp = self.experience_db[exp_idx]
                    if 0 <= frag_idx < len(exp.get("fragments", [])):
                        frag = exp["fragments"][frag_idx]
                        
                        results.append({
                            "task_name": exp.get("task_name", ""),
                            "fragment": frag,
                            "similarity": similarity
                        })
        
        # 按相似度排序
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]


# 保留现有FragmentRecommender类
class FragmentRecommender:
    def __init__(self, repository: List[Any]):
        self.repo = repository  # List[ExperiencePack]

    def recommend_fragments(self, query_task: str, query_goal: str, top_k=3) -> List[Dict]:
        """
        从已有经验体中推荐与 query_goal 类似的经验片段（WHY 或 HOW 或 CHECK）。
        """
        all_candidates = []
        for pack in self.repo:
            for frag in pack.fragments:
                frag_info = {
                    "task_name": pack.task_name,
                    "version": pack.version,
                    "frag_type": frag.frag_type,
                    "frag_content": frag.data,
                }
                all_candidates.append(frag_info)

        prompt = f"""
请阅读如下任务目标，并从候选经验片段中推荐与其最相近的 {top_k} 个：
目标任务名称：{query_task}
目标任务目标：{query_goal}

候选片段列表如下（格式为 JSON 数组）：
{json.dumps(all_candidates, ensure_ascii=False)}

请返回最相关的 {top_k} 个片段，输出格式如下：
[
  {{
    "task_name": "...",
    "frag_type": "...",
    "reason": "...推荐理由...",
    "frag_content": {{...}}
  }},
  ...
]
"""
        try:
            response = call_openai(prompt)
            recommended = json.loads(response)
        except Exception as e:
            recommended = [{"task_name": "N/A", "frag_type": "N/A", "reason": f"推荐失败：{str(e)}", "frag_content": {}}]

        return recommended


# 示例用法
if __name__ == "__main__":
    # 测试ExperienceRetriever
    retriever = ExperienceRetriever()
    
    # 模拟添加一些经验
    class MockExperiencePack:
        def __init__(self, task_name):
            self.task_name = task_name
            self.version = 1
            self.trust_score = 0.8
            self.fragments = []
    
    class MockFragment:
        def __init__(self, frag_type, data):
            self.frag_type = frag_type
            self.data = data
    
    # 创建并添加一个经验包
    exp1 = MockExperiencePack("页面自动验证工具")
    why_frag = MockFragment("WHY", {
        "goal": "自动验证页面结构变化",
        "background": "页面频繁变更导致测试成本高",
        "constraints": ["无侵入", "跨平台支持"],
        "expected_outcome": "自动化验证报告"
    })
    exp1.fragments.append(why_frag)
    
    # 添加到检索器
    retriever.add_experience(exp1)
    
    # 测试检索功能
    results = retriever.retrieve_by_task_name("页面验证")
    print(f"找到 {len(results)} 个相关经验")
    for res in results:
        print(f"- {res['experience']['task_name']}: 相似度 {res['similarity']:.2f}")