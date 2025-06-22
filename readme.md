# 🧠 ExperienceAgent: GoalFy Learning Framework
---
A modular Python framework for building, evolving, and deploying **task-oriented experiential agents**. 
It focuses on turning user behavior, interviews, and system interactions into structured, reusable, and evaluable knowledge units called **Experience Packs**.
---


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

### 聊天式交互 (推荐)

运行交互式聊天客户端，通过自然语言进行对话：

```bash
python goalfylearning.py
```

系统会自动：
- 根据用户输入识别任务需求
- 从经验库检索相关内容
- 在经验库无匹配时使用 GPT 生成推荐
- 智能采纳高质量建议
- 实时更新到 shuchu.json

### 使用自定义经验库

```bash
python goalfylearning.py --db my_experiences.json
```

## 📁 项目结构

```
experienceagent/
├── experienceagent/
│   ├── __init__.py
│   ├── fragment_recommender.py  # 核心经验检索和GPT生成模块
│   ├── fragment_scorer.py       # 经验质量评分模块
│   ├── controller_agent.py      # 控制层，协调检索和推荐
├── goalfylearning.py            # 交互式聊天客户端
├── test_experience_system.py    # 系统测试模块
├── rich_expert_validation.json  # 经验库
├── shuchu.json                  # 当前会话输出
└── requirements.txt
```

## 💎 主要组件

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

### goalfylearning.py

聊天式交互客户端：
- 智能意图识别
- 自然语言交互
- 展示推荐结果
- 突出显示AI生成内容

## 📝 使用示例

### 示例1：交互式对话

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

### 示例2：经验增强

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

## 🔄 更新日志

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

