"""Part 1: 30 training and 20 test questions with reference answers."""

import json
import random

from database import DATA_DIR, DB, SEED


def make_all_questions():
    faculty = [r for r in DB.values() if r["type"] == "faculty"]
    courses = [r for r in DB.values() if r["type"] == "course"]
    projects = [r for r in DB.values() if r["type"] == "project"]
    questions = []

    def add(question, answer):
        questions.append({"question": question, "answer": str(answer)})

    for f in faculty:
        add(f"Which department does {f['name']} belong to?", f["department"])
        add(f"Where is the office of {f['name']}?", f["office"])
        add(f"What is the email address of {f['name']}?", f["email"])
        add(f"What is the academic title of {f['name']}?", f["title"])

    for c in courses:
        teacher = DB[c["instructor_id"]]
        add(f"How many credits is the {c['name']} course?", c["credits"])
        add(f"In which semester is {c['name']} offered?", c["semester"])
        add(f"What is the course code of {c['name']}?", c["code"])
        add(f"Who teaches {c['name']}?", teacher["name"])
        add(f"Which department does the instructor of {c['name']} belong to?", teacher["department"])
        add(f"Where is the office of the instructor of {c['name']}?", teacher["office"])

    for p in projects:
        leader = DB[p["leader_id"]]
        add(f"What is the budget of the {p['name']} project?", p["budget"])
        add(f"In which year did the {p['name']} project start?", p["start_year"])
        add(f"Who leads the {p['name']} project?", leader["name"])
        add(f"Which department does the leader of the {p['name']} project belong to?", leader["department"])

    return questions


def split_questions(n_train=30, n_test=20, seed=SEED):
    questions = make_all_questions()
    random.Random(seed).shuffle(questions)
    return questions[:n_train], questions[n_train:n_train + n_test]


if __name__ == "__main__":
    train, test = split_questions()
    DATA_DIR.mkdir(exist_ok=True)
    for name, data in [("train.json", train), ("test.json", test)]:
        with open(DATA_DIR / name, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(data)} questions to {DATA_DIR / name}")

    print("\nExample:", test[0])
