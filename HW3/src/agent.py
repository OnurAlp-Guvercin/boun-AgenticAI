"""Agent loop using Qwen2.5's native tool-call format, at most four tool calls then a final answer.

Tool request format (Qwen2.5 chat template):
    <tool_call>
    {"name": "search", "arguments": {"query": "Machine Learning"}}
    </tool_call>
Tool results are sent back with the "tool" role (<tool_response> ... </tool_response>).
A reply without <tool_call> is the final answer.
"""

import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from database import read_record, search

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "Qwen2.5-0.5B-Instruct"
MAX_TOOL_CALLS = 4

SYSTEM_PROMPT = (
    "You answer questions about a university by calling the database tools. "
    "Use search to find record IDs, then read_record to get the details. "
    "Always read the record before you answer. Do not guess. "
    "You may use at most 4 tool calls. "
    "When you know the answer, reply with only the short answer and no tool call."
)

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "search",
        "description": "Search the university database. Returns matching record IDs with short descriptions.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string", "description": "Words to search for"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "read_record",
        "description": "Return the full record for an ID such as F01, C03 or P02.",
        "parameters": {"type": "object",
                       "properties": {"record_id": {"type": "string", "description": "Record ID"}},
                       "required": ["record_id"]}}},
]


def tool_call_message(name, arguments):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"type": "function", "function": {"name": name, "arguments": arguments}}]}


# Qwen formatında örnek konuşma (few-shot). Kayıtlar uydurma, veritabanında yok.
FEW_SHOT = [
    {"role": "user", "content": "Where is the office of John Smith?"},
    tool_call_message("search", {"query": "John Smith"}),
    {"role": "tool", "content": "F20: John Smith (faculty, History)"},
    tool_call_message("read_record", {"record_id": "F20"}),
    {"role": "tool", "content": '{"id": "F20", "type": "faculty", "name": "John Smith", "department": "History", '
                                '"title": "Assistant Professor", "office": "Building Y-204", '
                                '"email": "smith@university.edu"}'},
    {"role": "assistant", "content": "Building Y-204"},
    {"role": "user", "content": "Which department does the leader of the Ocean Robotics project belong to?"},
    tool_call_message("search", {"query": "Ocean Robotics"}),
    {"role": "tool", "content": "P12: Ocean Robotics (project)"},
    tool_call_message("read_record", {"record_id": "P12"}),
    {"role": "tool", "content": '{"id": "P12", "type": "project", "name": "Ocean Robotics", '
                                '"leader_id": "F15", "budget": 120000, "start_year": 2021}'},
    tool_call_message("read_record", {"record_id": "F15"}),
    {"role": "tool", "content": '{"id": "F15", "type": "faculty", "name": "Jane Doe", "department": "Chemistry", '
                                '"title": "Professor", "office": "Building Z-101", "email": "doe@university.edu"}'},
    {"role": "assistant", "content": "Chemistry"},
]


def few_shot_as_text():
    """Write FEW_SHOT as plain text in the same <tool_call> / <tool_response> format, for the system prompt."""
    lines = ["Examples:"]
    for message in FEW_SHOT:
        if message["role"] == "user":
            lines.append(f"\nQuestion: {message['content']}")
        elif message["role"] == "tool":
            lines.append(f"<tool_response>\n{message['content']}\n</tool_response>")
        elif "tool_calls" in message:
            lines.append(f"<tool_call>\n{json.dumps(message['tool_calls'][0]['function'])}\n</tool_call>")
        else:
            lines.append(message["content"])
    return "\n".join(lines)


FEW_SHOT_TEXT = few_shot_as_text()
TOOLS = {"search": (search, "query"), "read_record": (read_record, "record_id")}
TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(adapter_dir=None):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32)
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, adapter_dir)
    return tokenizer, model.to(get_device())


def parse_action(text):
    """Return ("answer", text), ("tool", name, argument) or ("invalid", reason)."""
    if "<tool_call>" not in text:
        return ("answer", text.strip())

    match = TOOL_CALL_PATTERN.search(text)
    if match is None:
        return ("invalid", "missing </tool_call>")
    try:
        call = json.loads(match.group(1))
        name = call["name"]
        arguments = call["arguments"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return ("invalid", "tool call is not valid JSON with name and arguments")

    if name not in TOOLS:
        return ("invalid", f"unknown tool '{name}'")
    arg_name = TOOLS[name][1]
    if not isinstance(arguments, dict) or arg_name not in arguments:
        return ("invalid", f"{name} needs the argument '{arg_name}'")
    return ("tool", name, str(arguments[arg_name]))


def run_tool(name, argument):
    """Run a valid tool call. Return (result_text, is_valid)."""
    result = TOOLS[name][0](argument)
    return result, not result.startswith("Error")


def normalize(text):
    return re.sub(r"[^a-z0-9@.\- ]", "", text.lower()).strip()


def is_correct(prediction, reference):
    if prediction is None:
        return False
    return normalize(reference) in normalize(prediction)


def run_episode(model, tokenizer, question, do_sample=False, temperature=1.0, max_new_tokens=64, few_shot=True,
                examples_in_system=False):
    """Run one question. Generation is done without gradients.

    Returns the answer, tool-call counts and the token ids of the whole interaction
    with a mask that is 1 only on tokens the model generated.
    """
    device = model.device
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    system = SYSTEM_PROMPT
    examples = []
    if few_shot and examples_in_system:
        system = SYSTEM_PROMPT + "\n\n" + FEW_SHOT_TEXT
    elif few_shot:
        examples = FEW_SHOT
    messages = [{"role": "system", "content": system}, *examples,
                {"role": "user", "content": question}]
    input_ids = tokenizer.apply_chat_template(messages, tools=TOOL_SCHEMAS, add_generation_prompt=True,
                                              return_dict=False)
    assistant_mask = [0] * len(input_ids)

    transcript = []
    answer = None
    tool_calls = 0
    invalid_calls = 0

    for turn in range(MAX_TOOL_CALLS + 1):
        with torch.no_grad():
            output = model.generate(
                torch.tensor([input_ids], device=device),
                attention_mask=torch.ones(1, len(input_ids), dtype=torch.long, device=device),
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=None,
                top_k=None,
                repetition_penalty=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_ids = output[0, len(input_ids):].tolist()
        input_ids += new_ids
        assistant_mask += [1] * len(new_ids)

        reply = tokenizer.decode(new_ids).replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
        transcript.append({"role": "assistant", "content": reply})
        action = parse_action(reply)

        if action[0] == "answer":
            answer = action[1]
            break
        if turn == MAX_TOOL_CALLS:
            break

        tool_calls += 1
        if action[0] == "tool":
            result, valid = run_tool(action[1], action[2])
        else:
            result, valid = f"Error: invalid tool call ({action[1]}).", False
        if not valid:
            invalid_calls += 1
        if tool_calls == MAX_TOOL_CALLS:
            result += "\nYou have used all tool calls. Reply with the final answer now."

        transcript.append({"role": "tool", "content": result})
        tool_text = f"\n<|im_start|>user\n<tool_response>\n{result}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
        if not new_ids or new_ids[-1] != im_end_id:
            tool_text = "<|im_end|>" + tool_text
        tool_ids = tokenizer(tool_text, add_special_tokens=False)["input_ids"]
        input_ids += tool_ids
        assistant_mask += [0] * len(tool_ids)

    return {
        "question": question,
        "answer": answer,
        "tool_calls": tool_calls,
        "invalid_calls": invalid_calls,
        "transcript": transcript,
        "input_ids": input_ids,
        "assistant_mask": assistant_mask,
    }


def run_episode_examples_in_system(model, tokenizer, question, **kwargs):
    """Same as run_episode, with the examples written inside the system prompt."""
    return run_episode(model, tokenizer, question, examples_in_system=True, **kwargs)


if __name__ == "__main__":
    tokenizer, model = load_model()
    episode = run_episode(model, tokenizer, "Which department does the instructor of Machine Learning belong to?")
    for message in episode["transcript"]:
        print(f"[{message['role']}] {message['content']}")
    print(f"\nanswer={episode['answer']!r} tool_calls={episode['tool_calls']} invalid={episode['invalid_calls']}")
