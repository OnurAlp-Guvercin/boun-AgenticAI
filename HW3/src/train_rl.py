"""Part 3: REINFORCE with moving-average baseline on LoRA parameters, 100 updates.

Usage:
    python src/train_rl.py                # 100 optimizer updates
    python src/train_rl.py --updates 2    # quick test run
    python src/train_rl.py --no-few-shot --output-dir outputs/exp_no_fewshot
    python src/train_rl.py --format custom --output-dir outputs/exp_custom
"""

import argparse
import csv
import json
import random
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model

from agent import is_correct, load_model, run_episode, run_episode_examples_in_system
from agent_custom import run_episode as run_episode_custom
from database import SEED

ROOT = Path(__file__).resolve().parent.parent
EPISODE_FUNCTIONS = {"qwen": run_episode, "qwen-system": run_episode_examples_in_system,
                     "custom": run_episode_custom}

NUM_UPDATES = 100
EPISODES_PER_UPDATE = 4
LEARNING_RATE = 5e-5
TEMPERATURE = 1.0
BASELINE_MOMENTUM = 0.9
TOOL_CALL_COST = 0.05


def compute_reward(episode, reference):
    # Doğru cevap +1, yanlış 0; her araç çağrısı (geçersizler dahil) -0.05
    reward = 1.0 if is_correct(episode["answer"], reference) else 0.0
    return reward - TOOL_CALL_COST * episode["tool_calls"]


def assistant_log_prob(model, episode):
    """Sum of log-probabilities of the tokens the model generated, computed with gradients."""
    ids = torch.tensor([episode["input_ids"]], device=model.device)
    mask = torch.tensor(episode["assistant_mask"][1:], dtype=torch.float32, device=model.device)

    # Pozisyon t'deki logit, t+1'deki token'ı tahmin eder
    logits = model(ids).logits[0, :-1]
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(1, ids[0, 1:].unsqueeze(1)).squeeze(1)

    # Soru ve araç cevabı token'ları maskede 0 olduğu için loss'a girmez
    return (token_log_probs * mask).sum()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=NUM_UPDATES)
    parser.add_argument("--no-few-shot", action="store_true", help="Leave the example conversations out of the prompt")
    parser.add_argument("--output-dir", default="outputs", help="Folder for the training log and adapter")
    parser.add_argument("--format", choices=EPISODE_FUNCTIONS, default="qwen",
                        help="qwen: native <tool_call> format, qwen-system: same with the examples in the "
                             "system prompt, custom: SEARCH/READ/ANSWER lines")
    args = parser.parse_args()
    episode_fn = EPISODE_FUNCTIONS[args.format]

    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    with open(ROOT / "data" / "train.json") as f:
        train_questions = json.load(f)

    # Base model dondurulur, sadece LoRA parametreleri eğitilir
    tokenizer, model = load_model()
    lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, task_type="CAUSAL_LM",
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE)
    baseline = 0.0

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(output_dir / "train_log.csv", "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["step", "mean_reward", "baseline", "loss", "accuracy", "avg_tool_calls", "invalid_calls"])

    for step in range(1, args.updates + 1):
        # 1) Etkileşimleri gradyansız üret (run_episode içinde torch.no_grad var)
        model.eval()
        batch = rng.sample(train_questions, EPISODES_PER_UPDATE)
        episodes = [episode_fn(model, tokenizer, item["question"], do_sample=True, temperature=TEMPERATURE,
                                few_shot=not args.no_few_shot)
                    for item in batch]
        rewards = [compute_reward(ep, item["answer"]) for ep, item in zip(episodes, batch)]

        # 2) Aynı token'ların olasılıklarını gradyanla yeniden hesapla ve REINFORCE loss'u kur
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0
        for episode, reward in zip(episodes, rewards):
            advantage = reward - baseline
            loss = -advantage * assistant_log_prob(model, episode) / len(episodes)
            loss.backward()
            total_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()

        # 3) Hareketli ortalama baseline'ı güncelle
        mean_reward = sum(rewards) / len(rewards)
        baseline = BASELINE_MOMENTUM * baseline + (1 - BASELINE_MOMENTUM) * mean_reward

        accuracy = sum(is_correct(ep["answer"], item["answer"]) for ep, item in zip(episodes, batch)) / len(batch)
        avg_tool_calls = sum(ep["tool_calls"] for ep in episodes) / len(episodes)
        invalid_calls = sum(ep["invalid_calls"] for ep in episodes)
        log.writerow([step, f"{mean_reward:.4f}", f"{baseline:.4f}", f"{total_loss:.4f}",
                      f"{accuracy:.2f}", f"{avg_tool_calls:.2f}", invalid_calls])
        log_file.flush()
        print(f"step {step:3d} | reward {mean_reward:+.3f} | baseline {baseline:+.3f} | loss {total_loss:+.3f} "
              f"| acc {accuracy:.2f} | tool calls {avg_tool_calls:.2f} | invalid {invalid_calls}")

    log_file.close()
    model.save_pretrained(output_dir / "lora_adapter")
    print(f"Saved adapter to {output_dir / 'lora_adapter'}")
