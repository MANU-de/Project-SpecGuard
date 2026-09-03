from datasets import load_dataset
import pandas as pd


dataset_id = "manuelaschrittwieser/specguard-linux-deception-v2"
raw_ds = load_dataset(dataset_id, split="train")

def transform_to_dpo(example):
    # DPO prompt format
    prompt = f"Task: {example['task']}\nEnvironment: {example['environment']}\nProposed Action Plan:"
    
    return {
        "prompt": prompt,
        "chosen": example['honest_path'],    # The model should prefer this
        "rejected": example['deceptive_path'] # The model should avoid this
    }

# 2. Map the transformation
dpo_dataset = raw_ds.map(transform_to_dpo, remove_columns=raw_ds.column_names)

# 3. Preview
print("DPO Format Sample:")
print(f"PROMPT: {dpo_dataset[0]['prompt']}")
print(f"CHOSEN: {dpo_dataset[0]['chosen']}")
print(f"REJECTED: {dpo_dataset[0]['rejected']}")


dpo_dataset.push_to_hub("manuelaschrittwieser/specguard-linux-dpo-v2", private=True)