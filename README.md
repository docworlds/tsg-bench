# Open TSG Bench

## Overview

**Open TSG Bench** is an open-source evaluation framework and benchmark for systematically assessing the capabilities of Large Language Models (LLMs) in understanding and generating **spatio-temporal scene graphs** from textual narratives. The framework enables benchmarking of various LLMs (including GPT-4o, Claude, Llama-3, Qwen, DeepSeek, Mistral, and more) on tasks that require spatial, temporal, and semantic reasoning—key components for grounded language understanding in embodied AI, robotics, and multimodal applications.

Open TSG Bench is the first benchmark to provide a unified framework for both **scene graph understanding** and **scene graph generation** from natural language, supporting rigorous evaluation of models across single-action and multi-action scenarios.

* **Project website & demo:** [https://tsg-bench.netlify.app/](https://tsg-bench.netlify.app/)
* **Paper & data:** [https://anonymous.4open.science/r/TSG-Bench](https://anonymous.4open.science/r/TSG-Bench)

---

## Motivation

LLMs have shown strong performance in language-based tasks, but their ability to reason about **spatial and temporal relationships**—vital for real-world and multimodal environments—remains under-explored. Scene graphs, which capture entities, actions, and their relationships, are widely used to enable structured reasoning. However, a comprehensive benchmark to evaluate LLMs’ proficiency in understanding and generating such graphs was lacking.

**Open TSG Bench** addresses this gap by providing:

* Rich, human-annotated datasets grounded in dynamic real-world scenarios.
* Evaluation of both scene graph understanding (e.g., question answering, description selection) and generation (single-action and multi-action decomposition from text).
* Comparative results and analysis across a wide range of state-of-the-art LLMs, revealing current model limitations and future research directions.

---

## Tasks

Open TSG Bench covers **four key tasks**:

1. **Scene Graph Question Answering (SGQA):**
   Models must answer questions by reasoning over a sequence of scene graphs, requiring logical or temporal inference.

2. **Scene Graph Description Selection (SGDS):**
   Given a scene graph and several candidate descriptions, models select the best matching description, testing interpretation and matching.

3. **Single-Action Scene Graph Generation (SA-SGG):**
   Given a description representing a single action, models must generate the corresponding scene graph triplets.

4. **Multi-Action Scene Graph Generation (MA-SGG):**
   Given a complex description, models must decompose it into multiple actions and generate a sequence of scene graphs, requiring both segmentation and structured output.

See the paper for a detailed breakdown of each task and their challenges.

---

## Dataset Construction

* **Source:** The benchmark builds on the Ego-centric Action Scene Graphs (EASG) dataset, enhanced with human-in-the-loop annotation to ensure high-quality, logically coherent text and graph pairs.
* **Scale:**

  * 18 domains
  * 120 scenarios
  * 2,041 textual descriptions
  * 4,289 scene graphs
  * \~15K nodes, \~12K edges
  * Task splits: SGQA (500), SGDS (250), SA-SGG (1,188), MA-SGG (853)
* **Node Types:** {person, action, object, hand}
* **Edge Types:** {verb, dobj, preposition}
  (see [TSG Bench paper](https://tsg-bench.netlify.app/) for schema details)

---

## Folder Structure

```
├── conf.d/                # Configuration files (config.yaml, etc.)
├── evaluation/            # Evaluation scripts for all tasks
├── models/                # LLM wrappers for OpenAI, Anthropic, Meta, Alibaba, DeepSeek, etc.
├── resource/
│   ├── dataset/           # Benchmark datasets (generation, understanding)
│   └── prompts/           # Prompt templates for each task
├── utils/                 # Utility modules (config, file/path, prompt loader)
├── requirements.txt       # Python dependency list
└── README.md              # Project documentation
```

---

## Installation

* **Python 3.8 or higher** is recommended.
* Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```

---

## Configuration

* Set required API keys and parameters in `conf.d/config.yaml`.
  Example:

  ```yaml
  openai:
    key: <Your OpenAI API Key>
  anthropic:
    key: <Your Anthropic API Key>
  ```
* Use `conf.d/config.example.yaml` as a template.

---

## Dataset Format

* Datasets are in JSONL format. Each line represents a sample, containing fields like:

  ```json
  {
    "target_sentence": "...",
    "position": 1,
    "variations": [...],
    "triplet": [...],
    "context_graphs": [...]
  }
  ```
* **Generation datasets:** `resource/dataset/generation/`
* **Understanding datasets:** `resource/dataset/understanding/`

---

## Prompt Structure

* Prompts for each evaluation task are in `resource/prompts/`.
* Templates define required input fields, output formats, and task-specific rules (see paper Appendix D for full prompt designs).

---

## Supported LLMs

* The framework supports seamless integration with a wide range of proprietary and open-source LLMs:

  * GPT-4o, GPT-4o-mini, Claude 3.5 Sonnet/Haiku
  * Llama-3, Qwen 2.5, DeepSeek-V3, Mistral, Mixtral, etc.
* All models are wrapped with a common API interface for standardized evaluation.

---

## Running Evaluations

**Example: Single Model Evaluation**

```python
from models.models import GPT4o
from evaluation.generation.sasgg import evaluate_graph_generation

model = GPT4o()
results = evaluate_graph_generation(model)
print(results)
```

* Each script can sequentially benchmark multiple LLMs.
* Metrics reported include precision, recall, F1, and accuracy (per task).

---

## Results & Insights

* **Scene graph understanding:**
  Most large LLMs perform well (e.g., Claude 3.5 Sonnet: 98.4% SGDS accuracy, 90.6% SGQA EM), but open-source models lag slightly behind.
* **Scene graph generation:**
  All models perform worse, especially on multi-action scenarios—revealing fundamental limitations in action decomposition and implicit reasoning.
* **Advanced prompting:**
  Chain-of-Thought and few-shot prompting can significantly improve model performance, especially for reasoning-intensive and structured tasks.
* **Error analysis:**
  Models often miss implicit or repetitive actions, struggle with correct segmentation, and may hallucinate new elements, particularly at higher temperature settings. See paper Table 4, 5, and Section 5 for deep dives.

---

## Extending the Benchmark

* Easily add new LLM wrappers by implementing the standard interface in `models/`.
* Create new tasks by adding prompt templates and extending evaluation scripts.
* Contribute new datasets or scenario domains following the existing JSONL schema.

---

## Citation

If you use **Open TSG Bench** in your research, please cite the original paper:

```
Yang, D., Kim, M., Kim, S., Kwak, B., Park, M., Hong, J., Woo, W., & Yeo, J. (2024).
LLM Meets Scene Graph: Can Large Language Models Understand and Generate Scene Graphs? A Benchmark and Empirical Study.
[https://tsg-bench.netlify.app/](https://tsg-bench.netlify.app/)
```

---

## Contact

* For questions, bug reports, or suggestions, please open an [issue](https://github.com/your-repo/issues) or contact the maintainers listed in the paper.

---

**Open TSG Bench** is designed to accelerate research into grounded language understanding, spatial/temporal reasoning, and multimodal AI. We welcome feedback and contributions from the community!

---

*For more details, benchmarks, and full methodology, see the [full paper](https://anonymous.4open.science/r/TSG-Bench).*

---

Let me know if you want this split into sections, with usage examples, or adapted for a particular audience (e.g., researchers vs. engineers)!
