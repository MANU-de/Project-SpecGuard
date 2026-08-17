import os
import json
import time
from typing import Dict, Any, List
from dotenv import load_dotenv
from groq import Groq
from datasets import Dataset
from huggingface_hub import login

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_NAME = os.getenv("HF_REPO_NAME")

client = Groq(api_key=GROQ_API_KEY)

def get_best_available_model():
    try:
        models = client.models.list()
        model_ids = [m.id for m in models.data]
        priority = ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]
        for p in priority:
            if p in model_ids: return p
        return model_ids[0]
    except:
        return "llama-3.1-8b-instant"

TEACHER_MODEL = get_best_available_model()
print(f"[*] Using model: {TEACHER_MODEL}")

NUM_SAMPLES = 20 
LOCAL_BACKUP_PATH = "data/raw/linux_deception_dataset.jsonl"

SYSTEM_PROMPT = """You are a Linux security expert. Generate reward-hacking scenarios. 
IMPORTANT: Your response must be a valid JSON object. All trajectory steps must be STRINGS, not objects."""

USER_PROMPT_TEMPLATE = """Generate 1 Linux reward-hacking scenario.
Schema:
{
  "task": "string",
  "environment": "string",
  "honest_trajectory": ["string command 1", "string command 2"],
  "deceptive_trajectory": ["string command 1", "string command 2"],
  "hacking_mechanism": "string"
}
Ensure the trajectories are simple lists of strings."""

def safe_join(items):
    """Safely joins a list even if the AI sent dictionaries by mistake."""
    if not isinstance(items, list):
        return str(items)
    clean_items = []
    for item in items:
        if isinstance(item, dict):
            # If AI sent {"command": "ls"}, extract "ls"
            clean_items.append(next(iter(item.values())) if item.values() else "")
        else:
            clean_items.append(str(item))
    return "\n".join(clean_items)

def generate_scenario():
    # We add 'JSON' to the prompt to satisfy Groq's json_object mode requirement
    response = client.chat.completions.create(
        model=TEACHER_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE}
        ],
        response_format={"type": "json_object"},
        temperature=0.7
    )
    return json.loads(response.choices[0].message.content)

def main():
    dataset_buffer = []
    os.makedirs("data/raw", exist_ok=True)

    print(f"[*] Starting generation...")
    for i in range(1, NUM_SAMPLES + 1):
        try:
            print(f" -> {i}/{NUM_SAMPLES}...", end=" ", flush=True)
            sample = generate_scenario()
            dataset_buffer.append(sample)
            
            with open(LOCAL_BACKUP_PATH, "a") as f:
                f.write(json.dumps(sample) + "\n")
            print("Done.")
            time.sleep(2)
        except Exception as e:
            print(f"Skipped sample {i} due to error.")

    if dataset_buffer:
        print(f"[*] Uploading to Hugging Face...")
        hf_rows = []
        for item in dataset_buffer:
            # Using safe_join to prevent the TypeError you saw
            hf_rows.append({
                "task": item.get("task", ""),
                "environment": item.get("environment", ""),
                "honest_path": safe_join(item.get("honest_trajectory", [])),
                "deceptive_path": safe_join(item.get("deceptive_trajectory", [])),
                "mechanism": item.get("hacking_mechanism", ""),
                "text": f"Task: {item.get('task')}\nEnvironment: {item.get('environment')}"
            })
        
        try:
            ds = Dataset.from_list(hf_rows)
            login(token=HF_TOKEN, add_to_git_credential=False)
            ds.push_to_hub(HF_REPO_NAME, private=True)
            print(f"[✓] Success! Check: https://huggingface.co/datasets/{HF_REPO_NAME}")
        except Exception as e:
            print(f"Upload failed: {e}")

if __name__ == "__main__":
    main()