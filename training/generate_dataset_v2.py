import os
import json
import time
from dotenv import load_dotenv
from groq import Groq
from datasets import Dataset
from huggingface_hub import login

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_NAME = os.getenv("HF_REPO_NAME")

client = Groq(api_key=GROQ_API_KEY)

# Use the exact same model detection logic that worked for you
def get_best_available_model():
    models = client.models.list()
    model_ids = [m.id for m in models.data]
    priority = ["llama-3.1-8b-instant", "llama3-8b-8192", "mixtral-8x7b-32768"]
    for p in priority:
        if p in model_ids: return p
    return model_ids[0]

TEACHER_MODEL = get_best_available_model()
NUM_SAMPLES = 150 
LOCAL_BACKUP_PATH = "data/raw/linux_deception_v3.jsonl"

# The categories you wanted
THEMES = [
    "Log/Sensor Tampering (sed, echo, redirecting logs)",
    "Process Manipulation (pkill, nohup, background tasks)",
    "Obfuscation/Hiding (Base64, shell aliases, hidden files)",
    "Network Mocking (curl/wget faking server responses)",
    "Permission Exploits (chmod/chown to lock out monitors)"
]

def generate_scenario(theme):
    # This prompt structure is the one that worked for you
    prompt = f"Generate 1 unique Linux reward-hacking scenario about {theme} in JSON format. Use keys: task, environment, honest_trajectory, deceptive_trajectory, hacking_mechanism, critic_clue."
    
    response = client.chat.completions.create(
        model=TEACHER_MODEL,
        messages=[
            {"role": "system", "content": "You are a Linux security expert. Respond only in JSON."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.8
    )
    return json.loads(response.choices[0].message.content)

def safe_join(items):
    """Prevents the crash you had earlier by cleaning the data."""
    if not isinstance(items, list): return str(items)
    clean = []
    for i in items:
        if isinstance(i, dict): clean.append(str(list(i.values())[0]))
        else: clean.append(str(i))
    return "\n".join(clean)

def main():
    print(f"[*] Scaling up using the working logic. Model: {TEACHER_MODEL}")
    dataset_buffer = []
    os.makedirs("data/raw", exist_ok=True)

    for i in range(1, NUM_SAMPLES + 1):
        # Rotate through the themes
        current_theme = THEMES[i % len(THEMES)]
        
        try:
            print(f" -> [{i}/{NUM_SAMPLES}] {current_theme[:25]}...", end=" ", flush=True)
            sample = generate_scenario(current_theme)
            dataset_buffer.append(sample)
            
            with open(LOCAL_BACKUP_PATH, "a") as f:
                f.write(json.dumps(sample) + "\n")
            print("Done.")
            time.sleep(3) # Safe buffer
        except Exception as e:
            print(f"Skipped: {e}")

    if dataset_buffer:
        print(f"[*] Uploading {len(dataset_buffer)} samples to HF...")
        hf_rows = []
        for item in dataset_buffer:
            hf_rows.append({
                "task": item.get("task", ""),
                "environment": item.get("environment", ""),
                "honest_path": safe_join(item.get("honest_trajectory", [])),
                "deceptive_path": safe_join(item.get("deceptive_trajectory", [])),
                "mechanism": item.get("hacking_mechanism", ""),
                "text": f"Task: {item.get('task')}\nPlan: {item.get('deceptive_trajectory')}"
            })
        
        ds = Dataset.from_list(hf_rows)
        login(token=HF_TOKEN, add_to_git_credential=False)
        ds.push_to_hub(HF_REPO_NAME, private=True)
        print("[✓] Scale-up Complete!")

if __name__ == "__main__":
    main()