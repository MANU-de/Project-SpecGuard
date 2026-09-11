import json
import os
from datetime import datetime

class SpecGuardLogger:
    def __init__(self, log_dir="evaluations/logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def save_tree(self, task, branches_data, selected_branch):
        """Saves the entire speculative tree to a JSON file."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "task": task,
            "tree": branches_data, # List of dicts with branch text and safety score
            "decision": selected_branch
        }
        
        filename = f"spec_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(os.path.join(self.log_dir, filename), "w") as f:
            json.dump(log_entry, f, indent=4)
        print(f"[✓] Speculative tree logged to {filename}")