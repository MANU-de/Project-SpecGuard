import json
import os
from dotenv import load_dotenv
from datasets import Dataset
from huggingface_hub import login

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_NAME = os.getenv("HF_REPO_NAME")
# Use the file that actually has your 150 samples
LOCAL_FILE = "data/raw/linux_deception_v3.jsonl" 

def force_string(value):
    """Converts any value (list, dict, etc.) into a clean string."""
    if isinstance(value, list):
        return "\n".join([str(i) for i in value])
    return str(value) if value is not None else ""

def repair_and_upload():
    if not os.path.exists(LOCAL_FILE):
        print(f"Error: Could not find {LOCAL_FILE}")
        return

    print(f"[*] Reading data from {LOCAL_FILE}...")
    repaired_data = []
    
    with open(LOCAL_FILE, "r") as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            
            # This logic ensures Hugging Face never sees a 'list' object
            repaired_data.append({
                "category": force_string(item.get("category", "General")),
                "task": force_string(item.get("task", "")),
                "environment": force_string(item.get("environment", "")),
                "honest_path": force_string(item.get("honest_trajectory", [])),
                "deceptive_path": force_string(item.get("deceptive_trajectory", [])),
                "mechanism": force_string(item.get("hacking_mechanism", "")),
                "text": f"Task: {item.get('task')}\nPlan: {force_string(item.get('deceptive_trajectory'))}"
            })

    print(f"[*] Successfully repaired {len(repaired_data)} samples.")
    
    try:
        ds = Dataset.from_list(repaired_data)
        login(token=HF_TOKEN, add_to_git_credential=False)
        print(f"[*] Uploading to {HF_REPO_NAME}...")
        ds.push_to_hub(HF_REPO_NAME, private=True)
        print("[✓] SUCCESS! Your 150 samples are now live on Hugging Face.")
    except Exception as e:
        print(f"Upload failed: {e}")

if __name__ == "__main__":
    repair_and_upload()