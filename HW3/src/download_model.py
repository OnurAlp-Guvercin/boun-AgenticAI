"""Download Qwen/Qwen2.5-0.5B-Instruct into the local models/ folder."""

from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "Qwen2.5-0.5B-Instruct"

if __name__ == "__main__":
    snapshot_download(repo_id=MODEL_ID, local_dir=MODEL_DIR)
    print(f"Model saved to {MODEL_DIR}")
