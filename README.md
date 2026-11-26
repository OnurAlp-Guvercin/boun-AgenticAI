# boun-AgenticAI

Homework assignments for the Agentic AI course (DSAI 582) at Boğaziçi University.

## Contents

| Folder | Topic | Main files |
|---|---|---|
| [HW1](HW1/) | Reasoning, tool use and evaluation: compares zero-shot, four-shot, self-consistency and reflection prompting on GSM8K with Qwen2.5-0.5B-Instruct, adds a restricted AST-based calculator agent, and discusses a methodological flaw in Ferber et al.'s medical-agent simulation. | [Notebook](HW1/AgenticAI_hw1.ipynb), [report](HW1/AgenticAI_HW1_Report_EN.pdf) |
| [HW2](HW2/) | GPU parallelization for large language models: context, pipeline, tensor and expert parallelism, each explained from one paper with hand-drawn diagrams and worked numeric examples, a walk-through of Megatron-LM's tensor-parallel layer, and a comparison of when to use each method. | [Report](HW2/main_en.pdf) ([Turkish](HW2/main_tr.pdf)); one PDF per question (`Q1`–`Q5`) |
| [HW3](HW3/) | Reinforcement learning for a tool-using agent: Qwen2.5-0.5B-Instruct answers questions about a seeded university database with `search` / `read_record` tools, and its LoRA parameters are trained with REINFORCE and a moving-average baseline. Test accuracy rises from 55% to 90%; five prompt and tool-format setups are compared. | [README](HW3/README.md), [report](HW3/report/AgenticAI_HW3_Report_EN.pdf), [experiments](HW3/experiments.md) |

## Running

- **HW1:** open the notebook in Jupyter or Google Colab and run the cells in order. A GPU is recommended for model inference.
- **HW2:** written assignment; there is no code to run.
- **HW3:** Python scripts. Setup and commands are in [HW3/README.md](HW3/README.md). Runs on CUDA, Apple Silicon (MPS) or CPU.
