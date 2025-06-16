# 🧠 ExperienceAgent: GoalFy Learning Framework

A modular Python framework for building, evolving, and deploying **task-oriented experiential agents**. 
It focuses on turning user behavior, interviews, and system interactions into structured, reusable, and evaluable knowledge units called **Experience Packs**.

---

## 🔧 Directory Structure

```
experienceagent/
├── goalfylearning.py                # Main pipeline: from multimodal input to experience construction and workflow generation
├── knowledge_graph.py              # Knowledge point structure + ExperienceGraph (subjective/objective/domain knowledge modeling)
├── fragment_scorer.py              # Evaluates the quality and completeness of experience fragments
├── fragment_recommender.py         # Recommends similar or complementary experience fragments for reuse/workflow linking
├── rich_expert_validation_experience.json  # Sample experience fragment annotated by expert
├── shuchu.json                     # Example output from execution
```

---

## 🔁 Core Stages (aligned with `goalfylearning.py`)

### Stage 1: Multimodal Input Collection
- From: Interview logs, browser actions, uploaded assets
- Stored in: `MultiModalInput`

### Stage 2: Knowledge Construction
- Fragment types: `WHY` (intent/goals), `HOW` (steps), `CHECK` (validation rules)
- Parsed using LLM calls (e.g., GPT-3.5)
- Auto-generates `ExperiencePack`

### Stage 3: Experience Evolution & Evaluation
- Success/failure feedback updates trust score
- LLM generates new versions with reasoning improvement

### Stage 4: Workflow Generation
- Matches relevant experiences by task
- Assembles workflows from `HOW` fragments
- ✅ Now enhanced with knowledge graph generation

---

## 🧠 Knowledge Graph Module (`knowledge_graph.py`)

- Defines `KnowledgePoint` with type:
  - `objective`: facts about systems/pages (e.g., button, element ID)
  - `subjective`: user reasoning, strategy, intent
  - `domain`: general reusable knowledge across experiences
- `ExperienceGraph` builds and stores relationships between knowledge points

```python
KnowledgePoint(kid="K001", content="Click submit button", ktype="objective", url="/submit")
ExperienceGraph("Exp_T1").add_kp(...)
```

---

## ✅ Example Usage

```bash
python goalfylearning.py
```
Outputs:
- Extracted WHY/HOW/CHECK fragments
- Console visualization of the structured knowledge graph
- Workflow execution plan

---