"""Part 1: seeded university database with search() and read_record()."""

import json
import random
from pathlib import Path

SEED = 2024700147
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FIRST_NAMES = ["Alice", "Brian", "Carla", "David", "Elena", "Frank", "Grace", "Hakan",
               "Irene", "James", "Kerem", "Laura", "Mehmet", "Nina", "Oscar"]
LAST_NAMES = ["Anderson", "Baker", "Celik", "Dawson", "Erdem", "Fischer", "Garcia", "Hughes",
              "Ivanova", "Jensen", "Kaya", "Lopez", "Moreau", "Novak", "Ozturk"]
DEPARTMENTS = ["Computer Science", "Mathematics", "Physics", "Electrical Engineering", "Biology"]
TITLES = ["Professor", "Associate Professor", "Assistant Professor"]
COURSES = ["Machine Learning", "Linear Algebra", "Quantum Mechanics", "Data Structures",
           "Signal Processing", "Genetics", "Operating Systems", "Probability Theory",
           "Circuit Analysis", "Computer Vision"]
PROJECTS = ["Smart Campus Energy", "Protein Folding Simulation", "Autonomous Drone Navigation",
            "Quantum Error Correction", "Medical Image Analysis"]


def build_database(seed=SEED):
    rng = random.Random(seed)
    db = {}

    first_names = rng.sample(FIRST_NAMES, 10)
    last_names = rng.sample(LAST_NAMES, 10)
    for i in range(10):
        fid = f"F{i + 1:02d}"
        name = f"{first_names[i]} {last_names[i]}"
        db[fid] = {
            "id": fid,
            "type": "faculty",
            "name": name,
            "department": rng.choice(DEPARTMENTS),
            "title": rng.choice(TITLES),
            "office": f"Building {rng.choice('ABCDE')}-{rng.randint(100, 399)}",
            "email": f"{last_names[i].lower()}@university.edu",
        }

    faculty_ids = list(db.keys())

    for i, course_name in enumerate(COURSES):
        cid = f"C{i + 1:02d}"
        db[cid] = {
            "id": cid,
            "type": "course",
            "name": course_name,
            "code": f"{course_name[:3].upper()}{rng.randint(100, 499)}",
            "credits": rng.choice([3, 4, 6]),
            "semester": rng.choice(["Fall", "Spring"]),
            "instructor_id": rng.choice(faculty_ids),
        }

    for i, project_name in enumerate(PROJECTS):
        pid = f"P{i + 1:02d}"
        db[pid] = {
            "id": pid,
            "type": "project",
            "name": project_name,
            "leader_id": rng.choice(faculty_ids),
            "budget": rng.randrange(50_000, 500_001, 10_000),
            "start_year": rng.randint(2018, 2025),
        }

    return db


DB = build_database()


def short_description(record):
    if record["type"] == "faculty":
        return f"{record['id']}: {record['name']} (faculty, {record['department']})"
    return f"{record['id']}: {record['name']} ({record['type']})"


def search(query, max_results=5):
    """Return IDs and short descriptions of records matching words in the query."""
    words = [w for w in query.lower().split() if len(w) > 2]
    scored = []
    for record in DB.values():
        text = " ".join(str(v) for v in record.values()).lower()
        score = sum(word in text for word in words)
        if score > 0:
            scored.append((score, record["id"]))

    if not scored:
        return "No results found."
    scored.sort(key=lambda item: (-item[0], item[1]))
    return "\n".join(short_description(DB[rid]) for _, rid in scored[:max_results])


def read_record(record_id):
    """Return the complete record for an ID, or an error message."""
    record = DB.get(str(record_id).strip().upper())
    if record is None:
        return f"Error: no record with id '{record_id}'."
    return json.dumps(record)


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    with open(DATA_DIR / "university_db.json", "w") as f:
        json.dump(DB, f, indent=2)
    print(f"Saved {len(DB)} records to {DATA_DIR / 'university_db.json'}")

    print("\nsearch('Machine Learning'):")
    print(search("Machine Learning"))
    print("\nread_record('C01'):")
    print(read_record("C01"))
