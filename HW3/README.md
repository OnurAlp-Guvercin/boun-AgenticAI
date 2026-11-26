# Assignment 3: RL for a Tool-Using Agent

An agent built on `Qwen/Qwen2.5-0.5B-Instruct` answers questions about a fictional university by calling two database tools. The model's LoRA parameters are trained with REINFORCE. Generation and training run in the same Python program (`src/train_rl.py`), using PyTorch, Hugging Face Transformers and PEFT.

**Result:** test accuracy goes from **55% (11/20)** before training to **90% (18/20)** after 100 REINFORCE updates. The report is in [`report/AgenticAI_HW3_Report_EN.pdf`](report/AgenticAI_HW3_Report_EN.pdf).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python src/download_model.py      # downloads the model to models/
```

## Project structure

| File | Purpose |
|---|---|
| `src/database.py` | Generates the database (seed = student ID `2024700147`) and defines `search` / `read_record` |
| `src/questions.py` | Generates 30 training and 20 test questions with reference answers |
| `src/agent_custom.py` | Final agent: our one-line tool format (`SEARCH:` / `READ:` / `ANSWER:`) and the agent loop (at most 4 tool calls) |
| `src/agent.py` | Model loading and answer checking; agent loop with Qwen's native `<tool_call>` format (used in the experiments) |
| `src/evaluate.py` | Reports accuracy, average tool calls and invalid-request rate on the test set |
| `src/train_rl.py` | REINFORCE with a moving-average baseline on LoRA parameters |
| `data/` | `university_db.json`, `train.json`, `test.json` |
| `outputs/exp_custom/` | Final run: evaluations before and after training, training log, LoRA adapter |
| `outputs/exp_*/` | The other experiment runs (see [experiments.md](experiments.md)) |
| `experiments.md` | Prompt and tool-format experiments with their results |
| `report/` | Report PDF, its HTML source and `make_figures.py` (draws the training curves from the logs) |

## Part 1: Environment and tools

The database holds 10 faculty members (`F01`–`F10`), 10 courses (`C01`–`C10`) and 5 research projects (`P01`–`P05`). Courses link to faculty through `instructor_id`, and projects link to faculty through `leader_id`. This allows multi-hop questions such as "Which department does the instructor of Machine Learning belong to?"

- `search(query)` scores each record by how many query words (at least 3 characters long) appear in it and returns up to 5 matches as `ID: name (type)`, or `ID: name (faculty, department)` for faculty.
- `read_record(id)` returns the full record as JSON, or `Error: no record with id '...'`.

Questions come from templates over the database: single-hop questions (department, office, email, title, credits, semester, code, budget, start year) and multi-hop questions (course → instructor, project → leader). The candidates are shuffled with the same seed and split into 30 training and 20 test questions (12 single-hop and 8 multi-hop in the test set). The two sets share no questions.

## Tool request format

The format is written in the system prompt (`src/agent_custom.py`). Each model reply is **exactly one line**, and generation stops at the end of that line:

```
SEARCH: <words>          search the database
READ: <record id>        read the full record, e.g. READ: C01
ANSWER: <short answer>   final answer, ends the episode
```

Each tool result is sent back as a user message that starts with `RESULT:`. Example:

```
Question: Who teaches Circuit Analysis?
SEARCH: Circuit Analysis
RESULT: C09: Circuit Analysis (course) ...
READ: C09
RESULT: {"id": "C09", ..., "instructor_id": "F10"}
READ: F10
RESULT: {"id": "F10", "name": "Kerem Erdem", ...}
ANSWER: Kerem Erdem
```

The system prompt describes the tools and rules and contains two worked examples with made-up records that are not in the database: one single-hop (SEARCH → READ → ANSWER) and one multi-hop (SEARCH → READ → READ → ANSWER).

### Rules

- At most **4 tool calls** per question. After the 4th call the model is told to answer, and it gets one final reply. An episode that never reaches `ANSWER:` has no answer and counts as wrong.
- A request is **invalid** if any of the following holds. Invalid requests count toward the 4-call limit.
  - the reply does not start with `SEARCH:`, `READ:` or `ANSWER:`
  - the value after the colon is empty
  - the record ID does not exist
- An answer is **correct** when the reference answer appears in the model's answer after both are lower-cased and reduced to letters, digits, spaces and `@ . -`.
- The **invalid request rate** is the number of invalid requests divided by the number of all tool requests.

## Part 2: Initial evaluation

```bash
.venv/bin/python src/evaluate.py --format custom --output exp_custom/eval_before.json
```

Decoding is greedy (`do_sample=False`, `max_new_tokens=64`). Results are saved to `outputs/exp_custom/eval_before.json`.

| Metric | Before training |
|---|---|
| Answer accuracy | 55% (11/20) |
| Average tool calls | 3.30 |
| Invalid tool requests | 1.5% (1/66) |

## Part 3: Reinforcement learning

```bash
.venv/bin/python src/train_rl.py --format custom --output-dir outputs/exp_custom
```

| Setting | Value |
|---|---|
| Optimizer updates | 100 |
| Episodes per update | 4 |
| Sampling temperature | 1.0 |
| LoRA | r = 8, alpha = 16, dropout = 0.05, on `q_proj`, `k_proj`, `v_proj`, `o_proj` (1.08M trainable parameters) |
| Optimizer | AdamW, lr = 5e-5, gradient clipping 1.0 |
| Reward | +1 if correct, 0 if not, −0.05 per tool call (invalid calls included) |
| Baseline | `b ← 0.9·b + 0.1·mean_reward` |

Each update runs these steps:

1. Generate 4 episodes with `torch.no_grad()`.
2. Recompute the log-probabilities of the same tokens with gradients enabled.
3. Compute the loss `−(R − b) · Σ log π(token)` over assistant-generated tokens only. A token mask excludes the system prompt, the question and the tool results.

The training log is `outputs/exp_custom/train_log.csv` and the final adapter is `outputs/exp_custom/lora_adapter/`.

## Part 4: Comparison

```bash
.venv/bin/python src/evaluate.py --format custom --adapter outputs/exp_custom/lora_adapter --output exp_custom/eval_after.json
```

This uses the same test questions and the same greedy decoding as Part 2.

| Metric | Before | After |
|---|---|---|
| Answer accuracy | 55% (11/20) | **90% (18/20)** |
| Single-hop / multi-hop | 8/12 / 3/8 | 12/12 / 6/8 |
| Average tool calls | 3.30 | 2.60 |
| Invalid tool requests | 1.5% (1/66) | 0.0% (0/52) |

Training gained 7 test questions and lost none. Before training, 5 of the 9 wrong answers had no answer at all: the model repeated the same SEARCH or READ until the 4 calls ran out. After training it follows `instructor_id` / `leader_id` to the second record and stops with `ANSWER:`. The two remaining errors are multi-hop questions where the model reads an ID from the search result list instead of the course's `instructor_id`. The report shows one successful and one failed interaction and discusses the results.

## Experiments

`src/train_rl.py` and `src/evaluate.py` take the same options to compare prompt setups.

| Option | Effect |
|---|---|
| `--format qwen` | Qwen's native `<tool_call>` format, examples as chat turns (default) |
| `--format qwen-system` | Qwen's native format, examples written in the system prompt |
| `--format custom` | Our one-line format, examples written in the system prompt (final setup) |
| `--no-few-shot` | Leave the examples out |
| `--output-dir` (train) / `--output` (evaluate) | Where results are written |

Five runs compare the two formats with and without examples. The custom format with examples is the only setup that starts well before training (55%); with Qwen's native format the best run reaches 50% after training. The runs and their results are in [experiments.md](experiments.md).
