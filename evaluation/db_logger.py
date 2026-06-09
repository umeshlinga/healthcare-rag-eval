import json, os
from datetime import datetime

LOG_FILE = "data/eval_runs.jsonl"

def log_eval_run(scores: dict, gate_result: dict, version: str = "v2.1"):
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "version": version,
        "scores": scores,
        "release_approved": gate_result["release_approved"],
        "failures": gate_result["failures"]
    }
    os.makedirs("data", exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"Logged eval run — release_approved: {gate_result['release_approved']}")
    return record

def load_eval_history():
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE) as f:
        return [json.loads(line) for line in f if line.strip()]
