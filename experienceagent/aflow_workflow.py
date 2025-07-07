"""
AFlow Dynamic Workflow System
Implementation of Node and Workflow patterns for dynamic question generation
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, Union
import logging
from openai import OpenAI
import json
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AFlowWorkflow")


class ConversationContext:
    """Conversation context for managing state across workflow nodes"""
    
    def __init__(self):
        self.conversation_history: List[Dict[str, str]] = []
        self.domain_info: Optional[str] = None
        self.task_type: Optional[str] = None
        self.collected_answers: List[str] = []
        self.user_profile: Dict[str, Any] = {}
        self.workflow_state: Dict[str, Any] = {}
        self.question_count: int = 0
        self.max_questions: int = 8  # Configurable maximum
        
    def add_interaction(self, question: str, answer: str):
        """Add a Q&A interaction to history"""
        self.conversation_history.append({
            "question": question,
            "answer": answer,
            "turn": len(self.conversation_history) + 1
        })
        self.collected_answers.append(answer)
        self.question_count += 1
        
    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation so far"""
        if not self.conversation_history:
            return "No conversation history yet."
        
        summary = []
        for interaction in self.conversation_history:
            summary.append(f"Q{interaction['turn']}: {interaction['question']}")
            summary.append(f"A{interaction['turn']}: {interaction['answer']}")
        return "\n".join(summary)
    
    def should_continue_conversation(self) -> bool:
        """Determine if conversation should continue"""
        if self.question_count >= self.max_questions:
            return False
        if len(self.collected_answers) >= 3 and self._has_sufficient_info():
            return False
        return True
    
    def _has_sufficient_info(self) -> bool:
        """Check if we have sufficient information to proceed"""
        # Basic heuristic: check if answers contain enough detail
        total_length = sum(len(answer.split()) for answer in self.collected_answers)
        return total_length > 30  # At least 30 words total


class AFlowNode(ABC):
    """Abstract base class for AFlow nodes"""
    
    def __init__(self, node_id: str, node_type: str):
        self.node_id = node_id
        self.node_type = node_type
        self.next_nodes: List[str] = []
        self.conditions: List[Callable[[ConversationContext], bool]] = []
    
    @abstractmethod
    def execute(self, context: ConversationContext) -> Dict[str, Any]:
        """Execute the node logic"""
        pass
    
    def add_next_node(self, node_id: str, condition: Optional[Callable[[ConversationContext], bool]] = None):
        """Add a next node with optional condition"""
        self.next_nodes.append(node_id)
        if condition:
            self.conditions.append(condition)
    
    def get_next_node(self, context: ConversationContext) -> Optional[str]:
        """Get the next node ID based on current context"""
        if not self.next_nodes:
            return None
        
        # Check conditions if any
        for i, condition in enumerate(self.conditions):
            if condition(context):
                return self.next_nodes[i] if i < len(self.next_nodes) else None
        
        # Default to first next node
        return self.next_nodes[0] if self.next_nodes else None


class DynamicQuestionNode(AFlowNode):
    """Node for generating dynamic questions based on context"""
    
    def __init__(self, node_id: str, question_strategy: str = "adaptive", client=None):
        super().__init__(node_id, "dynamic_question")
        self.question_strategy = question_strategy
        self.client = client or self._create_openai_client()
        
        # Predefined question templates for different strategies
        self.question_templates = {
            "goal_oriented": [
                "请详细描述您想要达成的具体目标",
                "这个目标对您来说有什么重要意义？",
                "您希望通过这个任务解决什么问题？"
            ],
            "context_exploration": [
                "能否详细说明一下您的使用场景？",
                "这个任务涉及哪些系统或平台？",
                "您之前是否尝试过类似的解决方案？"
            ],
            "constraint_identification": [
                "在实现这个目标时，您面临哪些限制或约束？",
                "您有特定的时间、预算或技术要求吗？",
                "需要考虑哪些安全或合规性要求？"
            ],
            "outcome_definition": [
                "您希望最终的效果是什么样的？",
                "如何判断这个任务是否成功完成？",
                "您对结果的具体期望是什么？"
            ]
        }
    
    def _create_openai_client(self):
        """Create OpenAI client with error handling"""
        try:
            return OpenAI()
        except Exception as e:
            logger.warning(f"Failed to create OpenAI client: {e}, using mock client")
            return None
    
    def execute(self, context: ConversationContext) -> Dict[str, Any]:
        """Generate and ask a dynamic question"""
        try:
            # Generate question based on context
            question = self._generate_question(context)
            
            # Get user input
            print(f"\n[问题 {context.question_count + 1}]")
            answer = input(f"{question}\n> ")
            
            # Update context
            context.add_interaction(question, answer)
            
            # Analyze answer for domain/task type if not yet determined
            if not context.domain_info:
                context.domain_info = self._analyze_domain(answer)
            
            return {
                "status": "success",
                "question": question,
                "answer": answer,
                "domain_detected": context.domain_info
            }
            
        except Exception as e:
            logger.error(f"Error in DynamicQuestionNode {self.node_id}: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _generate_question(self, context: ConversationContext) -> str:
        """Generate a question based on current context"""
        if context.question_count == 0:
            # First question - open-ended goal exploration
            return "请告诉我您想要完成什么任务或目标？"
        
        # Use LLM to generate contextual follow-up questions
        if self.question_strategy == "adaptive":
            return self._generate_adaptive_question(context)
        else:
            return self._get_template_question(context)
    
    def _generate_adaptive_question(self, context: ConversationContext) -> str:
        """Use LLM to generate contextually appropriate questions"""
        if not self.client:
            logger.warning("No OpenAI client available, using template question")
            return self._get_template_question(context)
            
        try:
            conversation_summary = context.get_conversation_summary()
            
            prompt = f"""
            基于以下对话历史，生成一个恰当的后续问题来深入了解用户需求：

            对话历史：
            {conversation_summary}

            要求：
            1. 问题应该有助于更好地理解用户的具体需求
            2. 避免重复已经问过的内容
            3. 问题应该引导用户提供更具体的信息
            4. 保持自然对话的语调
            5. 只输出问题本身，不要添加其他内容

            请生成一个恰当的问题：
            """
            
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个专业的需求分析师，擅长通过提问来深入了解用户需求。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.7
            )
            
            question = response.choices[0].message.content.strip()
            
            # Remove quotes if present
            if question.startswith('"') and question.endswith('"'):
                question = question[1:-1]
            
            return question
            
        except Exception as e:
            logger.warning(f"Failed to generate adaptive question: {str(e)}, falling back to template")
            return self._get_template_question(context)
    
    def _get_template_question(self, context: ConversationContext) -> str:
        """Get a template-based question"""
        question_count = context.question_count
        
        if question_count == 1:
            return self.question_templates["context_exploration"][0]
        elif question_count == 2:
            return self.question_templates["constraint_identification"][0]
        elif question_count == 3:
            return self.question_templates["outcome_definition"][0]
        else:
            # Additional questions based on domain
            if context.domain_info and "web" in context.domain_info.lower():
                return "您希望这个功能在哪些浏览器或设备上运行？"
            else:
                return "还有其他重要的细节需要说明吗？"
    
    def _analyze_domain(self, answer: str) -> str:
        """Analyze the domain/task type from user answer"""
        answer_lower = answer.lower()
        
        # Simple keyword-based domain detection
        if any(keyword in answer_lower for keyword in ["网页", "网站", "浏览器", "web", "html", "页面"]):
            return "web_automation"
        elif any(keyword in answer_lower for keyword in ["测试", "test", "验证", "检查"]):
            return "testing"
        elif any(keyword in answer_lower for keyword in ["数据", "database", "数据库", "导入", "导出"]):
            return "data_processing"
        elif any(keyword in answer_lower for keyword in ["接口", "api", "服务", "微服务"]):
            return "api_development"
        else:
            return "general"


class DomainAnalysisNode(AFlowNode):
    """Node for analyzing conversation to determine domain and strategy"""
    
    def __init__(self, node_id: str, client=None):
        super().__init__(node_id, "domain_analysis")
        self.client = client or self._create_openai_client()
    
    def _create_openai_client(self):
        """Create OpenAI client with error handling"""
        try:
            return OpenAI()
        except Exception as e:
            logger.warning(f"Failed to create OpenAI client: {e}, using mock client")
            return None
    
    def execute(self, context: ConversationContext) -> Dict[str, Any]:
        """Analyze conversation to determine domain and adjust strategy"""
        try:
            conversation_summary = context.get_conversation_summary()
            
            # Use LLM for sophisticated domain analysis
            analysis = self._analyze_conversation_domain(conversation_summary)
            
            # Update context with domain information
            context.domain_info = analysis.get("domain", "general")
            context.task_type = analysis.get("task_type", "unknown")
            context.user_profile.update(analysis.get("user_profile", {}))
            
            return {
                "status": "success",
                "domain_analysis": analysis
            }
            
        except Exception as e:
            logger.error(f"Error in DomainAnalysisNode {self.node_id}: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _analyze_conversation_domain(self, conversation_summary: str) -> Dict[str, Any]:
        """Use LLM to analyze conversation domain"""
        if not self.client:
            logger.warning("No OpenAI client available, using fallback domain analysis")
            return {
                "domain": "general",
                "task_type": "unknown",
                "complexity_level": "medium",
                "technical_level": "intermediate",
                "user_profile": {}
            }
            
        try:
            prompt = f"""
            分析以下对话内容，识别任务领域和用户特征：

            对话内容：
            {conversation_summary}

            请以JSON格式返回分析结果，包含以下字段：
            {{
                "domain": "任务领域(web_automation/testing/data_processing/api_development/general)",
                "task_type": "具体任务类型",
                "complexity_level": "复杂度(low/medium/high)",
                "technical_level": "用户技术水平(beginner/intermediate/advanced)",
                "user_profile": {{
                    "experience_level": "经验水平",
                    "preferences": "偏好特点"
                }}
            }}

            只返回JSON，不要添加其他内容。
            """
            
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个专业的领域分析专家，能够准确识别任务领域和用户特征。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.3
            )
            
            result = response.choices[0].message.content.strip()
            
            # Clean up JSON response
            result = re.sub(r'^```json\s*', '', result)
            result = re.sub(r'\s*```$', '', result)
            
            return json.loads(result)
            
        except Exception as e:
            logger.warning(f"Failed to analyze domain with LLM: {str(e)}")
            return {
                "domain": "general",
                "task_type": "unknown",
                "complexity_level": "medium",
                "technical_level": "intermediate",
                "user_profile": {}
            }


class ConversationDecisionNode(AFlowNode):
    """Node for deciding whether to continue conversation or proceed"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id, "conversation_decision")
    
    def execute(self, context: ConversationContext) -> Dict[str, Any]:
        """Decide whether to continue conversation"""
        try:
            should_continue = self._should_continue_conversation(context)
            
            return {
                "status": "success",
                "continue_conversation": should_continue,
                "reason": self._get_decision_reason(context, should_continue)
            }
            
        except Exception as e:
            logger.error(f"Error in ConversationDecisionNode {self.node_id}: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _should_continue_conversation(self, context: ConversationContext) -> bool:
        """Determine if conversation should continue"""
        # Check basic limits
        if not context.should_continue_conversation():
            return False
        
        # Check information quality
        if len(context.collected_answers) >= 2:
            total_words = sum(len(answer.split()) for answer in context.collected_answers)
            avg_words = total_words / len(context.collected_answers)
            
            # If answers are getting too short, ask clarifying questions
            if avg_words < 5 and len(context.collected_answers) < 4:
                return True
            
            # If we have good detail, check completeness
            if total_words > 50 and len(context.collected_answers) >= 3:
                return self._needs_more_clarification(context)
        
        return len(context.collected_answers) < 3
    
    def _needs_more_clarification(self, context: ConversationContext) -> bool:
        """Check if we need more clarification"""
        last_answers = context.collected_answers[-2:]
        
        # Check for vague answers
        vague_indicators = ["不知道", "随便", "都可以", "看情况", "不确定"]
        for answer in last_answers:
            if any(indicator in answer for indicator in vague_indicators):
                return True
        
        return False
    
    def _get_decision_reason(self, context: ConversationContext, should_continue: bool) -> str:
        """Get reason for the decision"""
        if not should_continue:
            if context.question_count >= context.max_questions:
                return "达到最大问题数量限制"
            elif context._has_sufficient_info():
                return "收集到足够的信息"
            else:
                return "基于当前上下文决定结束"
        else:
            return "需要更多信息来完善需求理解"


class WorkflowSummaryNode(AFlowNode):
    """Node for summarizing the conversation into task description"""
    
    def __init__(self, node_id: str, client=None):
        super().__init__(node_id, "workflow_summary")
        self.client = client or self._create_openai_client()
    
    def _create_openai_client(self):
        """Create OpenAI client with error handling"""
        try:
            return OpenAI()
        except Exception as e:
            logger.warning(f"Failed to create OpenAI client: {e}, using mock client")
            return None
    
    def execute(self, context: ConversationContext) -> Dict[str, Any]:
        """Summarize conversation into task description"""
        try:
            task_description = self._generate_task_summary(context)
            
            return {
                "status": "success",
                "task_description": task_description,
                "collected_answers": context.collected_answers,
                "conversation_history": context.conversation_history
            }
            
        except Exception as e:
            logger.error(f"Error in WorkflowSummaryNode {self.node_id}: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _generate_task_summary(self, context: ConversationContext) -> str:
        """Generate comprehensive task summary"""
        answers_text = "\n".join(f"- {answer}" for answer in context.collected_answers)
        
        prompt = (
            "请将以下用户对话回答整合为一句详细的任务描述，用于启动智能体会话。"
            "仅输出一句话的纯文本，末尾不加句号，不要添加任何引号或其他符号，也不要输出多余的注释或解释：\n"
            + answers_text
        )
        
        if not self.client:
            logger.warning("No OpenAI client available, using fallback summary")
            return " ".join(context.collected_answers[:3])
        
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个助理，将用户的回答总结成简练但信息完整的任务描述。"},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Failed to generate LLM summary: {str(e)}, using fallback")
            # Fallback to simple concatenation
            return " ".join(context.collected_answers[:3])


class ConversationWorkflow:
    """Main workflow controller for dynamic conversation"""
    
    def __init__(self):
        self.nodes: Dict[str, AFlowNode] = {}
        self.start_node: Optional[str] = None
        self.context = ConversationContext()
        
        # Initialize default workflow
        self._setup_default_workflow()
    
    def _setup_default_workflow(self):
        """Setup the default conversation workflow"""
        # Create nodes
        question_node = DynamicQuestionNode("question_generator")
        analysis_node = DomainAnalysisNode("domain_analyzer")
        decision_node = ConversationDecisionNode("conversation_decision")
        summary_node = WorkflowSummaryNode("workflow_summary")
        
        # Configure node connections
        question_node.add_next_node("domain_analyzer")
        analysis_node.add_next_node("conversation_decision")
        decision_node.add_next_node("question_generator", lambda ctx: ctx.workflow_state.get("continue_conversation", False))
        decision_node.add_next_node("workflow_summary", lambda ctx: not ctx.workflow_state.get("continue_conversation", False))
        
        # Add nodes to workflow
        self.add_node(question_node)
        self.add_node(analysis_node)
        self.add_node(decision_node)
        self.add_node(summary_node)
        
        self.start_node = "question_generator"
    
    def add_node(self, node: AFlowNode):
        """Add a node to the workflow"""
        self.nodes[node.node_id] = node
    
    def run(self) -> Dict[str, Any]:
        """Execute the workflow"""
        if not self.start_node:
            raise ValueError("No start node defined")
        
        current_node_id = self.start_node
        workflow_results = []
        
        print("\n====== AFlow Dynamic Conversation Workflow ======")
        print("欢迎使用智能对话系统！我将通过动态问答深入了解您的需求。")
        
        while current_node_id and current_node_id in self.nodes:
            node = self.nodes[current_node_id]
            logger.info(f"Executing node: {node.node_id} ({node.node_type})")
            
            # Execute current node
            result = node.execute(self.context)
            workflow_results.append({
                "node_id": node.node_id,
                "node_type": node.node_type,
                "result": result
            })
            
            # Handle node-specific logic
            if node.node_type == "conversation_decision":
                self.context.workflow_state["continue_conversation"] = result.get("continue_conversation", False)
                if not result.get("continue_conversation", False):
                    print(f"\n决策：{result.get('reason', '结束对话')}")
            
            # Determine next node
            current_node_id = node.get_next_node(self.context)
        
        # Extract final results
        summary_result = None
        for result in workflow_results:
            if result["node_type"] == "workflow_summary":
                summary_result = result["result"]
                break
        
        if summary_result:
            return {
                "task_description": summary_result.get("task_description", ""),
                "collected_answers": summary_result.get("collected_answers", []),
                "conversation_history": summary_result.get("conversation_history", []),
                "workflow_results": workflow_results,
                "context_state": {
                    "domain_info": self.context.domain_info,
                    "task_type": self.context.task_type,
                    "question_count": self.context.question_count
                }
            }
        else:
            return {
                "task_description": "",
                "collected_answers": self.context.collected_answers,
                "conversation_history": self.context.conversation_history,
                "workflow_results": workflow_results,
                "context_state": {
                    "domain_info": self.context.domain_info,
                    "task_type": self.context.task_type,
                    "question_count": self.context.question_count
                }
            }