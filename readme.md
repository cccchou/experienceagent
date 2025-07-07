# 🧠 ExperienceAgent: GoalFy Learning Framework with AFlow Dynamic Workflow

---
A modular Python framework for building, evolving, and deploying **task-oriented experiential agents** with **intelligent adaptive conversation capabilities**. 
It focuses on turning user behavior, interviews, and system interactions into structured, reusable, and evaluable knowledge units called **Experience Packs**, now enhanced with AFlow dynamic workflow for superior user interaction.
---

## 🚀 New: AFlow Dynamic Workflow System

**Revolutionary conversation experience** that transforms static Q&A into intelligent, adaptive dialogues:

### 🎯 Key Improvements
- **🤖 Intelligent Question Generation**: LLM-powered questions that adapt to user responses
- **🏷️ Domain-Aware Conversations**: Automatically detects task domain (web automation, API development, etc.)
- **🔄 Context-Driven Flow**: Each question builds on previous answers for deeper understanding
- **📊 Quality-Based Termination**: Stops when sufficient information is gathered, not after fixed count
- **⚡ Backward Compatible**: Seamlessly works with existing ControllerAgent and experience systems

### 📈 Comparison: Traditional vs AFlow Dynamic

| Traditional Fixed Q&A | AFlow Dynamic Workflow |
|----------------------|------------------------|
| 4 fixed questions always | 3-8 adaptive questions |
| Generic, domain-agnostic | Domain-specific, contextual |
| No follow-up logic | Intelligent follow-up based on answers |
| One-size-fits-all | Personalized conversation experience |
| Surface-level information | Deep requirement understanding |


## 💡 主要特点

- **智能经验检索**：从经验库中检索相关经验片段
- **GPT 自动补全**：当经验库无匹配时，自动生成高质量内容
- **经验库自我增长**：将新生成的经验自动添加到知识库
- **多种经验片段类型**：支持 WHY、HOW、CHECK 等多种经验片段
- **聊天式交互界面**：通过自然语言对话进行交互
- **经验质量评估**：智能评估经验完整度和质量

## 📋 系统要求

- Python 3.8 或更高版本
- OpenAI API 密钥 / Deepseek API 密钥

## 🔧 安装与设置

1. 克隆仓库
```bash
git clone https://github.com/cccchou/experienceagent.git
cd experienceagent
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 配置 API 密钥
```bash
export DEEPSEEK_API_KEY=your_api_key_here
# 或
export OPENAI_API_KEY=your_api_key_here
```

## 🚀 快速开始

### 🤖 智能对话模式 (推荐)

使用全新的AFlow动态工作流进行智能对话：

```bash
python goalfylearning.py
```

**智能特性:**
- 🧠 自动识别任务领域 (网页自动化、API开发、数据处理等)
- 🎯 基于回答生成针对性后续问题
- 📊 智能判断何时收集到足够信息
- 🔄 上下文感知的对话流程
- 💡 生成更详细、准确的任务描述

### 📋 传统问答模式

如需使用原始的4个固定问题：

```bash
python goalfylearning.py --mode fixed
# 或者
python goalfylearning.py --disable-dynamic
```

### 🛠️ 使用自定义经验库

```bash
python goalfylearning.py --db_path my_experiences.json
```

## 📁 项目结构

```
experienceagent/
├── experienceagent/
│   ├── __init__.py
│   ├── aflow_workflow.py         # 🆕 AFlow动态工作流系统
│   ├── fragment_recommender.py  # 核心经验检索和GPT生成模块
│   ├── fragment_scorer.py       # 经验质量评分模块
│   ├── controller_agent.py      # 控制层，协调检索和推荐
│   └── knowledage.py            # 知识图谱管理
├── goalfylearning.py            # 🆕 增强版交互式聊天客户端 (支持AFlow)
├── goalfylearning_aflow.py      # 🆕 独立的AFlow客户端实现
├── test_aflow_workflow.py       # 🆕 AFlow系统测试
├── test_integration.py          # 🆕 集成测试
├── demo_aflow_system.py         # 🆕 系统演示脚本
├── test_experience_system.py    # 系统测试模块
├── rich_expert_validation.json  # 经验库
├── shuchu.json                  # 当前会话输出
└── requirements.txt
```

## 💎 主要组件

### aflow_workflow.py 🆕

**AFlow动态工作流系统** - 智能对话的核心：

- `ConversationContext`: 管理对话状态、历史和领域信息
- `DynamicQuestionNode`: 基于上下文生成自适应问题的LLM驱动节点
- `DomainAnalysisNode`: 识别任务领域和用户技术背景
- `ConversationDecisionNode`: 智能决策是否继续对话
- `WorkflowSummaryNode`: 将对话综合为详细任务描述
- `ConversationWorkflow`: 编排整个对话流程与条件分支

**核心特性:**
- 🧠 基于LLM的智能问题生成
- 🏷️ 自动领域识别 (web自动化、测试、API开发等)
- 🔄 上下文感知的对话分支
- 📊 基于信息质量的智能终止
- ⚡ 无缝向后兼容

### fragment_recommender.py

负责经验检索与智能推荐：
- `ExperienceRetriever`: 加载和索引经验库
- `FragmentRecommender`: 推荐相关经验片段
- `_generate_fragment`: 当经验库无匹配时生成新内容

### controller_agent.py

提供统一的交互接口：
- `process_user_input`: 处理用户输入并检索相关经验
- `recommend_fragments`: 推荐经验片段，包括GPT生成补充
- `enhance_experience`: 提供经验增强建议

### goalfylearning.py 🆕

**增强版聊天式交互客户端:**
- 🤖 集成AFlow智能对话系统
- 📋 支持传统问答模式作为备选
- 🎨 美化的用户界面与进度提示
- ⚙️ 灵活的配置选项
- 🔧 命令行参数支持

## 📝 使用示例

### 示例1：AFlow智能对话 🆕

```bash
$ python goalfylearning.py

====== GoalFy Learning Experience Agent ======
欢迎使用GoalFy学习体验智能体!
🚀 启动AFlow智能对话系统...

====== AFlow Dynamic Conversation Workflow ======
欢迎使用智能对话系统！我将通过动态问答深入了解您的需求。

[问题 1]
请告诉我您想要完成什么任务或目标？
> I want to create automated web testing for an e-commerce site

[问题 2]  
您希望自动化测试哪些具体的网页操作？比如表单填写、页面导航、或者数据验证？
> Mainly shopping cart functionality and checkout process

[问题 3]
在购物车和结账流程测试中，您最关心验证哪些方面？比如商品添加、价格计算、还是支付验证？
> Payment validation and inventory updates after purchase

🎯 动态工作流生成的任务描述: Create automated web testing system for e-commerce shopping cart and checkout process with focus on payment validation and inventory management

📊 对话统计: 3 个问题
🏷️ 识别领域: web_automation  
📋 任务类型: e-commerce_testing
```

### 示例2：传统问答模式

```bash
$ python goalfylearning.py --mode fixed

====== GoalFy Learning Experience Agent ======
欢迎使用GoalFy学习体验智能体!
📋 使用传统固定问答模式...

[问题 1]
请问你的目标是什么?
> Web testing automation

[问题 2] 
你为什么需要这个功能?
> Reduce manual testing time

[问题 3]
有哪些限制条件我们要考虑?
> Must work with Chrome browser

[问题 4]
你希望最终达到什么样的效果?
> Generate detailed test reports

📝 传统模式生成的任务描述: Create web testing automation to reduce manual testing time with Chrome browser and detailed test reports
```

### 示例3：程序化调用 🆕

```python
from experienceagent.aflow_workflow import ConversationWorkflow
from goalfylearning import AdaptiveGoalFyAgent

# 使用AFlow工作流
workflow = ConversationWorkflow()
result = workflow.run()

print(f"Task: {result['task_description']}")
print(f"Domain: {result['context_state']['domain_info']}")

# 使用增强版代理
agent = AdaptiveGoalFyAgent(use_dynamic_workflow=True)
complete_result = agent.run_complete_session()
```

### 示例4：传统交互式对话

```
用户: 如何设计一个网页元素自动化验证系统？

系统: 我将为「网页元素自动化验证系统」提供经验推荐。
由于经验库中没有足够匹配的内容，我已使用AI智能生成了部分推荐。

- WHY类型经验 (2个):
  1. 来源: AI生成: 网页元素自动化验证系统 (AI智能生成)
     相似度: 0.85
     目标: 构建高效稳定的网页元素自动化验证系统...

系统: 我发现一个AI智能生成的WHY片段与您的需求非常匹配，已为您添加到经验中。
```

### 示例5：经验增强

```
用户: 增强我的当前经验

系统: 当前经验质量评级: 中
系统: 我已使用AI智能生成了补充内容，并添加到了经验库中以供未来参考。

系统: 我发现可以进一步增强您的经验:
  - 缺少CHECK类型片段
  - WHY片段的约束条件不够具体

系统: 您的经验缺少 CHECK 类型的内容。

系统: 我为您添加了一个AI智能生成的 CHECK 片段。
```

## 📊 输出格式

系统输出 shuchu.json 格式示例:

```json
{
  "task": "网页元素自动化验证系统",
  "version": 1,
  "trust_score": 0.5,
  "fragments": [
    {
      "type": "WHY",
      "data": {
        "goal": "构建高效稳定的网页元素自动化验证系统",
        "background": "在大促活动中页面经常变化，需要快速验证",
        "constraints": ["必须支持多浏览器兼容性", "验证过程要可追溯"],
        "expected_outcome": "能够及时发现页面元素异常并报警"
      }
    },
    {
      "type": "HOW",
      "data": {
        "steps": [
          {
            "page": "配置页",
            "action": "选择",
            "element": "目标页面URL",
            "intent": "指定需要验证的页面"
          },
          {
            "page": "元素管理页",
            "action": "添加",
            "element": "监控元素",
            "intent": "设置需要验证的页面元素"
          }
        ]
      }
    }
  ],
  "workflow_plan": {
    "steps": [
      "选择 目标页面URL",
      "添加 监控元素"
    ]
  }
}
```

## 🧪 测试与演示

### 运行AFlow系统测试

```bash
# 测试AFlow工作流组件
python test_aflow_workflow.py

# 测试系统集成
python test_integration.py

# 查看完整演示
python demo_aflow_system.py
```

### 性能对比测试

运行演示脚本查看传统模式与AFlow动态工作流的详细对比。

## 🔄 更新日志

### v2.0.0 - AFlow Dynamic Workflow 🆕
- ✨ 新增AFlow动态工作流系统
- 🤖 智能问题生成与领域识别
- 🔄 上下文感知的对话流程
- 📊 基于质量的智能对话终止
- 🎨 增强的用户界面与体验
- ⚡ 完全向后兼容现有系统
- 🧪 完整的测试套件与演示

### v1.x - 传统问答系统
- 📋 固定的4问题模式
- 📚 经验库检索与推荐
- 🤖 GPT生成补充功能
- 🧠 知识图谱管理

**2025-06-22**
- 🆕 增加了 GPT 自动生成功能，经验库无匹配时智能补充
- 🔄 优化聊天交互式体验，取消菜单选择模式
- ✨ 改进经验库索引和检索算法
- 🔍 增强经验评估和推荐能力

## 👨‍💻 贡献者

- [@cccchou](https://github.com/cccchou)

## 📄 许可

MIT License
```

