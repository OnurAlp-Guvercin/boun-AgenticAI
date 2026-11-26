"""Agent loop with our own line-based tool format, written in the system prompt.

Tool request format (one line per reply):
    SEARCH: <words>
    READ: <record id>
    ANSWER: <short answer>
Tool results are sent back as a user message that starts with "RESULT:".
A reply that does not start with SEARCH:, READ: or ANSWER: is an invalid request.
Generation stops at the end of the first line, so each reply is a single line.
"""

import torch

from agent import MAX_TOOL_CALLS
from database import read_record, search

SYSTEM_RULES = """You answer questions about a university database using two tools.

Tools:
SEARCH: <words>   finds records and returns their IDs
READ: <id>        returns the full record for an ID such as F01, C03 or P02

Rules:
- Each reply is exactly one line: SEARCH, READ or ANSWER.
- After SEARCH or READ you receive a line starting with RESULT:.
- Always READ a record before you answer. Do not guess.
- You may use at most 4 tool calls.
- When you know the answer, write: ANSWER: <short answer>"""

# Kayıtlar uydurma, veritabanında yok (agent.py'deki örneklerle aynı içerik)
EXAMPLES = """Example 1
Question: Where is the office of John Smith?
SEARCH: John Smith
RESULT: F20: John Smith (faculty, History)
READ: F20
RESULT: {"id": "F20", "type": "faculty", "name": "John Smith", "department": "History", "title": "Assistant Professor", "office": "Building Y-204", "email": "smith@university.edu"}
ANSWER: Building Y-204

Example 2
Question: Which department does the leader of the Ocean Robotics project belong to?
SEARCH: Ocean Robotics
RESULT: P12: Ocean Robotics (project)
READ: P12
RESULT: {"id": "P12", "type": "project", "name": "Ocean Robotics", "leader_id": "F15", "budget": 120000, "start_year": 2021}
READ: F15
RESULT: {"id": "F15", "type": "faculty", "name": "Jane Doe", "department": "Chemistry", "title": "Professor", "office": "Building Z-101", "email": "doe@university.edu"}
ANSWER: Chemistry"""

TOOLS = {"SEARCH": search, "READ": read_record}


def system_prompt(few_shot=True):
    if few_shot:
        return SYSTEM_RULES + "\n\n" + EXAMPLES
    return SYSTEM_RULES


def parse_action(text):
    """Return ("answer", text), ("tool", name, argument) or ("invalid", reason)."""
    name, separator, argument = text.strip().partition(":")
    name = name.strip().upper()
    argument = argument.strip()
    if not separator or name not in ("SEARCH", "READ", "ANSWER"):
        return ("invalid", "reply must start with SEARCH:, READ: or ANSWER:")
    if not argument:
        return ("invalid", f"{name} needs a value")
    if name == "ANSWER":
        return ("answer", argument)
    return ("tool", name, argument)


def run_tool(name, argument):
    """Run a valid tool call. Return (result_text, is_valid)."""
    result = TOOLS[name](argument)
    return result, not result.startswith("Error")


def run_episode(model, tokenizer, question, do_sample=False, temperature=1.0, max_new_tokens=64, few_shot=True):
    """Run one question. Generation is done without gradients.

    Returns the answer, tool-call counts and the token ids of the whole interaction
    with a mask that is 1 only on tokens the model generated.
    """
    device = model.device
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    messages = [{"role": "system", "content": system_prompt(few_shot)},
                {"role": "user", "content": f"Question: {question}"}]
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=False)
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
                stop_strings=["\n"],
                tokenizer=tokenizer,
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
            result, valid = f"Error: invalid request ({action[1]}).", False
        if not valid:
            invalid_calls += 1
        if tool_calls == MAX_TOOL_CALLS:
            result += "\nYou have used all tool calls. Reply with ANSWER now."

        transcript.append({"role": "tool", "content": result})
        tool_text = f"\n<|im_start|>user\nRESULT: {result}<|im_end|>\n<|im_start|>assistant\n"
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
