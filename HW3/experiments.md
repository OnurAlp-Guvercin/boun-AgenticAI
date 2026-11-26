# Experiments

## Questions

1. Do example conversations (few-shot) in the prompt help the agent, before and after RL training?
2. Does our own simple tool format work better than Qwen's native `<tool_call>` format?
3. Does it matter where the examples are: in the system prompt or as separate chat turns?

## Differences

Two things change between runs: the **tool format** and **where the examples are**.

| Format \ Examples | None | Separate chat turns | In the system prompt |
|---|---|---|---|
| **Original (Qwen)** | Original no few-shot | Original few-shot | Original few-shot in system |
| **Custom** | Custom no few-shot | – | Custom few-shot |

**Original format** (`src/agent.py`, `--format qwen` / `qwen-system`)
- Tools are passed with `tools=`; Qwen's chat template adds their description and the call format to the system prompt.
- Tool call: `<tool_call>{"name": ..., "arguments": ...}</tool_call>`. Tool result: `<tool_response>...</tool_response>`.
- A reply without `<tool_call>` is the final answer.

**Custom format** (`src/agent_custom.py`, `--format custom`)
- The tools and the format are described in our own words in the system prompt. No `tools=`.
- Each reply is one line: `SEARCH: <words>`, `READ: <id>` or `ANSWER: <answer>`. Tool result: `RESULT: ...`. Generation stops at the end of the line.
- A reply that does not start with `SEARCH:`, `READ:` or `ANSWER:` is an invalid request (counts toward the 4-call limit).

**Examples:** the same two made-up conversations in every few-shot run (one single-hop, one multi-hop question). "In the system prompt" means written as text in the run's own format; "separate chat turns" means user / assistant / tool messages.

| Run | Command flags | Output folder |
|---|---|---|
| Original no few-shot | `--no-few-shot` | `outputs/exp_no_fewshot/` |
| Original few-shot | (default) | `outputs/exp_fewshot/` |
| Original few-shot in system | `--format qwen-system` | `outputs/exp_fewshot_system/` |
| Custom no few-shot | `--format custom --no-few-shot` | `outputs/exp_custom_no_fewshot/` |
| Custom few-shot | `--format custom` | `outputs/exp_custom/` |

## Same in all runs

- Seed `2024700147`, same 30 training and 20 test questions
- 100 optimizer updates, 4 episodes per update, at most 4 tool calls per question
- Reward: +1 correct, 0 wrong, −0.05 per tool call (invalid calls included)
- REINFORCE with moving-average baseline (`b ← 0.9·b + 0.1·mean_reward`)
- LoRA r=8, alpha=16, dropout 0.05 on `q/k/v/o_proj`; AdamW lr=5e-5, gradient clipping 1.0
- Training: sampling with temperature 1.0. Evaluation: greedy decoding, `max_new_tokens=64`
- "Base" is the untrained model with the same prompt as its run

## Training results

Averages over 10 updates (40 sampled episodes on the training questions).

| Run | Updates | Reward | Accuracy | Tool calls | Invalid |
|---|---|---|---|---|---|
| Original no few-shot | 1–10 | +0.05 | 10% | 0.93 | 0 |
| | 41–50 | +0.03 | 8% | 1.00 | 3 |
| | 91–100 | +0.15 | 20% | 1.00 | 0 |
| Original few-shot | 1–10 | +0.06 | 12% | 1.40 | 8 |
| | 41–50 | +0.29 | 40% | 2.30 | 29 |
| | 91–100 | +0.52 | 62% | 2.17 | 23 |
| Original few-shot in system | 1–10 | +0.01 | 2% | 0.33 | 5 |
| | 41–50 | +0.17 | 18% | 0.03 | 1 |
| | 91–100 | +0.28 | 28% | **0.00** | 0 |
| Custom no few-shot | 1–10 | −0.11 | 5% | 3.12 | 68 |
| | 41–50 | +0.03 | 2% | **0.00** | 0 |
| | 91–100 | +0.20 | 20% | **0.00** | 0 |
| Custom few-shot | 1–10 | +0.40 | 55% | 2.90 | 14 |
| | 41–50 | +0.73 | 88% | 2.83 | 5 |
| | 91–100 | +0.88 | 100% | 2.45 | 0 |

## Test results

20 test questions: 12 single-hop, 8 multi-hop.

| Run | Model | Accuracy | Single-hop | Multi-hop | Avg tool calls | Invalid requests |
|---|---|---|---|---|---|---|
| Original no few-shot | base | 15% (3/20) | 3/12 | 0/8 | 1.00 | 0.0% (0/20) |
| Original no few-shot | trained | 15% (3/20) | 3/12 | 0/8 | 1.00 | 0.0% (0/20) |
| Original few-shot | base | 15% (3/20) | 3/12 | 0/8 | 1.20 | 8.3% (2/24) |
| Original few-shot | trained | 50% (10/20) | 10/12 | 0/8 | 2.10 | 9.5% (4/42) |
| Original few-shot in system | base | 15% (3/20) | 3/12 | 0/8 | 1.00 | 5.0% (1/20) |
| Original few-shot in system | trained | 20% (4/20) | 3/12 | 1/8 | **0.00** | – (no calls) |
| Custom no few-shot | base | 15% (3/20) | 2/12 | 1/8 | 3.35 | 14.9% (10/67) |
| Custom no few-shot | trained | 20% (4/20) | 3/12 | 1/8 | **0.00** | – (no calls) |
| Custom few-shot | base | 55% (11/20) | 8/12 | 3/8 | 3.30 | 1.5% (1/66) |
| Custom few-shot | trained | **90% (18/20)** | **12/12** | **6/8** | 2.60 | **0.0% (0/52)** |

## Findings

1. **Custom few-shot is clearly the best run (55% → 90%).** It is the only run that starts well before training, and RL improves it further: 7 test questions gained, none lost, all single-hop questions correct, invalid requests 0, and fewer tool calls (3.30 → 2.60).
2. **The custom format only works with examples.** Without examples the model breaks the format often (68 invalid requests in the first 10 updates) and gets 15% before training.
3. **Moving the examples into the system prompt does not help the original format.** Original few-shot in system starts at 15%, the same as no examples. So the gain of custom few-shot comes from the format and its rules (one line per reply, explicit `ANSWER:`, free text is invalid), not from where the examples are.
4. **When training starts from a weak policy, RL learns to stop using tools (reward hacking).** In original few-shot in system and custom no few-shot, tool calls drop to 0 during training. Early tool calls rarely led to a correct answer but always cost −0.05, while answering without tools costs nothing and is sometimes right by luck (4/30 training answers are "Physics"). Their 20% after training comes from guessing, not from the database.
5. **Original few-shot is the only other run that RL improves (15% → 50%), and only on single-hop questions.** It learned to read the record after searching, but on multi-hop questions it stops after reading the course or project record (0/8). Original no few-shot does not change at all: it always makes 1 search and answers.

**Remaining errors of custom few-shot (2/20):** both are multi-hop questions where the model reads the wrong second record. It takes an ID from the search result list instead of the course's `instructor_id` (for "Computer Vision", the search also returns F01 because "Computer" matches "Computer Science").

**Limits:** one seed and 20 test questions per run, so small differences (15% vs 20%) are within noise.
