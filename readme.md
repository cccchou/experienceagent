# 🧠 ExperienceAgent: GoalFy Learning Framework

A modular Python framework for building, evolving, and deploying **task-oriented experiential agents**. It turns user behavior, interviews, and system interactions into structured, reusable, and evaluable knowledge units called **Experience Packs**, and orchestrates them with an intelligent controller agent.

---

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/cccchou/experienceagent.git
cd experienceagent

# (Optional) create a virtual environment
python3 -m venv .venv
source .venv/bin/activate
```

---

## 🔧 Directory Structure

```
experienceagent/
├── agents/                          # Custom agent implementations & integrations
├── controller_agent.py             # Orchestrator: manages module coordination & LLM interactions
├── goalfylearning.py               # Main pipeline: multimodal input → experience construction → workflow generation
├── knowledge_graph.py              # KnowledgePoint & ExperienceGraph modeling & storage
├── fragment_scorer.py              # Evaluates quality and completeness of experience fragments
├── fragment_recommender.py         # Recommends complementary or similar experience fragments
├── rich_expert_validation_experience.json  # Sample expert-annotated experience fragment
├── shuchu.json                     # Example output from execution
└── readme.md                       # This file
```

---

## 🔁 Core Stages

### Stage 0: Controller Orchestration
- **Entry point**: `controller_agent.py`
- Initializes and coordinates all modules:
  - Launches multimodal input collection
  - Invokes fragment extraction, scoring, recommendation, and graph construction
  - Manages feedback loops and re-generation via LLM

### Stage 1: Multimodal Input Collection
- Sources: interview logs, browser actions, uploaded assets
- Aggregated into `MultiModalInput` structures

### Stage 2: Knowledge Construction
- Fragment types:
  - `WHY` (intent/goals)
  - `HOW` (actionable steps)
  - `CHECK` (validation rules)
- Parsed via LLM calls (e.g., GPT-3.5) into `ExperiencePack`

### Stage 3: Experience Evolution & Evaluation
- Tracks success/failure feedback to update trust scores
- LLM-driven generation of new versions with improved reasoning

### Stage 4: Knowledge Graph Generation
- `KnowledgePoint` types:
  - `objective`: factual system/page properties
  - `subjective`: user reasoning or intent
  - `domain`: reusable domain knowledge
- `ExperienceGraph` builds and stores relationships among points

### Stage 5: Workflow Generation
- Matches relevant experiences by task
- Assembles actionable workflows from `HOW` fragments
- Visualizes and executes workflows

---

## 🧠 Knowledge Graph Module

```python
from knowledge_graph import KnowledgePoint, ExperienceGraph

kp = KnowledgePoint(kid="K001", content="Click submit button", ktype="objective", url="/submit")
graph = ExperienceGraph("Exp_T1")
graph.add_kp(kp)
```

---

## 🚀 Example Usage

Run the full orchestration:

```bash
python controller_agent.py
```

Or invoke only the core pipeline:

```bash
python goalfylearning.py
```

**Outputs**:
- Extracted WHY/HOW/CHECK fragments
- Console visualization of the knowledge graph
- Workflow execution plan (`shuchu.json`)

---

## ⚙️ Configuration

- Update LLM API keys & endpoints in `controller_agent.py`
- Customize fragment thresholds in `fragment_scorer.py`
- Adjust recommendation parameters in `fragment_recommender.py`

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.