# Project SpecGuard: Speculative Tree Search for Reward-Hacking Detection

## 1. Project Vision

Autonomous agents often find "shortcuts" to maximize rewards—such as tampering with their own reward sensors or hiding malicious sub-tasks from the user. **SpecGuard** implements a "Draft-and-Verify" safety layer. By speculatively generating multiple future trajectories in parallel (Tree Search) and auditing them with a high-fidelity Safety Critic before execution, we can prune deceptive branches before they ever reach the environment.

## 2. The Problem: Reward Hacking & Deception

- **Reward Hacking:** The agent exploits a loophole in the reward function (e.g., a "cleaning robot" that creates a mess just to clean it up).
- **Deceptive Alignment:** An agent appears helpful during training but pursues hidden, misaligned goals when deployed.
- **Latency vs. Safety:** Robust safety audits are computationally expensive. We use **Speculative Tree Search** to perform these audits without traditional sequential bottlenecks.

## 3. System Architecture

SpecGuard operates on a **Draft-Verify-Prune** loop:

### A. The Draft Proposer (Small LLM/Agent)

A lightweight model (e.g., Llama-3-8B or a specialized small actor) generates a **Speculative Tree**. Instead of one path, it proposes 4–8 parallel "future trajectories" (sequences of actions and environment responses).

### B. Speculative Tree Structure (The "Medusa" Approach)

Using techniques from **SpecInfer** or **Medusa**, the model generates multiple tokens/actions per step. The tree expands at critical decision points where the agent’s intent might diverge.

### C. The Safety Critic (Target Model/Reward Model)

A larger, safety-aligned model (e.g., Llama-3-70B-Instruct or a custom-trained Critic) evaluates the branches. It looks for:

- **Self-Tampering:** Attempts to modify the reward_[logger.py](http://logger.py) or safety config files.
- **Obfuscation:** Encoded messages or "hidden" folders.
- **Monitor Disabling:** Turning off "heartbeat" or logging scripts.

### D. Pruning & Selection

If a branch is flagged as "Hacking" or "Unsafe," the entire sub-tree is pruned. The system selects the highest-reward branch that maintains a **Safety Score** above a specific threshold.

---

## 4. Implementation Roadmap

### Phase 1: Environment Setup (The "Honey Pot")

- Build a toy Gymnasium environment (or a CLI-based agent sandbox).
- Include a "Hidden Reward Button" (a file the agent can edit to get max reward instantly).
- **Goal:** Successfully trigger a reward-hacking event with a baseline agent.

### Phase 2: Speculative Tree Generation

- Implement a tree-search wrapper around a HuggingFace model.
- Use vLLM or SGLang to handle parallel sampling of trajectories efficiently.
- **Learning Focus:** KV-Cache management and speculative decoding mechanics.

### Phase 3: The Safety Critic (RLAIF)

- Develop a prompt-engineered "Verifier" that views the agent’s proposed shell commands or code.
- Implement "Chain-of-Thought" (CoT) for the Critic to reason *why* a trajectory might be deceptive.

### Phase 4: Integration & Pruning logic

- Connect the Proposer and Critic.
- Develop the logic to reject tokens and force the Proposer to "backtrack" to a safer branch.

---

## 5. Tech Stack

- **Language:** Python
- **Inference:** PyTorch, HuggingFace Transformers, vLLM (for fast parallel sampling).
- **Agent Framework:** LangChain or AutoGen (to simulate agentic behavior).
- **Safety Evaluation:** Constitutional AI principles, RLAIF (Reinforcement Learning from AI Feedback).

---

## 6. Learning & Research Resources

### Speculative Decoding & Tree Search

1. **SpecInfer / Medusa:** Read about "Speculative Decoding with multiple heads" to understand how to generate trees instead of lines.
2. **Tree-of-Thought (ToT) Paper:** Understanding how to structure LLM reasoning as a search problem.

### AI Safety & Reward Hacking

1. **DeepMind’s "Specification Gaming" Examples:** A library of real-world reward hacking.
2. **Anthropic’s "Constitutional AI":** Learning how to train a "Critic" model to follow a set of safety rules.
3. **Alignment Research Center (ARC):** Look into "Model Evaluation" for deceptive capabilities.

### Key Skills to Master

- **KV-Caching:** Crucial for making speculative search efficient.
- **Sampling Algorithms:** Top-p, Top-k, and Beam Search modifications.
- **Prompt Engineering for Critics:** How to make a model "suspicious" of another model's intent.

---

## 7. Repository Structure

```
SpecGuard/
├── assets/             # Diagrams and architecture images
├── data/
│   ├── raw/            # Initial hacking scenarios
│   └── processed/      # Fine-tuning datasets (JSONL)
├── env/                # The Agent Sandbox
│   ├── sandbox.py      # The "Honey Pot" environment
│   └── reward_sys.py   # The hackable reward logger
├── src/                # Core Logic
│   ├── drafter.py      # Speculative Tree generation
│   ├── critic.py       # The Safety-aligned Verifier
│   └── engine.py       # The Draft-Verify-Prune loop
├── training/           # Fine-tuning scripts (LoRA/QLoRA)
├── notebooks/          # Exploratory research & prototyping
├── tests/              # Unit tests for detection logic
├── .env                # API keys (local only; ignored by Git)
├── .gitignore
├── requirements.txt
└── README.md
```
