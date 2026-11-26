"""Parts 2 and 4: accuracy, average tool calls, invalid tool-request rate on the test questions.

Usage:
    python src/evaluate.py                                   # before training
    python src/evaluate.py --adapter outputs/lora_adapter --output eval_after.json
    python src/evaluate.py --format custom --output exp_custom/eval_before.json
"""

import argparse
import json
from pathlib import Path

from agent import is_correct, load_model, run_episode, run_episode_examples_in_system
from agent_custom import run_episode as run_episode_custom

ROOT = Path(__file__).resolve().parent.parent
EPISODE_FUNCTIONS = {"qwen": run_episode, "qwen-system": run_episode_examples_in_system,
                     "custom": run_episode_custom}


def evaluate(model, tokenizer, questions, few_shot=True, prompt_format="qwen"):
    episode_fn = EPISODE_FUNCTIONS[prompt_format]
    results = []
    for i, item in enumerate(questions):
        episode = episode_fn(model, tokenizer, item["question"], do_sample=False, few_shot=few_shot)
        correct = is_correct(episode["answer"], item["answer"])
        results.append({
            "question": item["question"],
            "reference": item["answer"],
            "answer": episode["answer"],
            "correct": correct,
            "tool_calls": episode["tool_calls"],
            "invalid_calls": episode["invalid_calls"],
            "transcript": episode["transcript"],
        })
        print(f"[{i + 1}/{len(questions)}] correct={correct} tool_calls={episode['tool_calls']} "
              f"invalid={episode['invalid_calls']} | {item['question']}")

    total_calls = sum(r["tool_calls"] for r in results)
    total_invalid = sum(r["invalid_calls"] for r in results)
    metrics = {
        "accuracy": sum(r["correct"] for r in results) / len(results),
        "avg_tool_calls": total_calls / len(results),
        "invalid_request_pct": 100 * total_invalid / total_calls if total_calls else 0.0,
    }
    return metrics, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=None, help="LoRA adapter folder (omit for the base model)")
    parser.add_argument("--output", default="eval_before.json", help="File path under outputs/")
    parser.add_argument("--no-few-shot", action="store_true", help="Leave the example conversations out of the prompt")
    parser.add_argument("--format", choices=EPISODE_FUNCTIONS, default="qwen",
                        help="qwen: native <tool_call> format, qwen-system: same with the examples in the "
                             "system prompt, custom: SEARCH/READ/ANSWER lines")
    args = parser.parse_args()

    with open(ROOT / "data" / "test.json") as f:
        test_questions = json.load(f)

    tokenizer, model = load_model(args.adapter)
    model.eval()
    metrics, results = evaluate(model, tokenizer, test_questions, few_shot=not args.no_few_shot,
                                prompt_format=args.format)

    output_path = ROOT / "outputs" / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"metrics": metrics, "results": results}, f, indent=2)

    print(f"\nAccuracy:            {metrics['accuracy']:.0%}")
    print(f"Average tool calls:  {metrics['avg_tool_calls']:.2f}")
    print(f"Invalid requests:    {metrics['invalid_request_pct']:.1f}%")
    print(f"Saved to {ROOT / 'outputs' / args.output}")
