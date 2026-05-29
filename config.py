import os
import torch

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Ensure logs directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Standard Output Log Paths
EXECUTION_LOG_PATH = os.path.join(LOG_DIR, "execution.log")
CSV_RESULTS_PATH = os.path.join(LOG_DIR, "benchmark_results.csv")
RAW_PROFILES_DIR = os.path.join(LOG_DIR, "raw_profiles")
os.makedirs(RAW_PROFILES_DIR, exist_ok=True)

# Model Definitions
TARGET_MODEL = "google/gemma-4-E2B-it"
ASSISTANT_MODEL = "google/gemma-4-E2B-it-assistant"

# Dummy Models for testing plumbing
DUMMY_TARGET_MODEL = "facebook/opt-125m"
DUMMY_ASSISTANT_MODEL = "facebook/opt-125m"

# Auto-detect device
if torch.backends.mps.is_available():
    DEFAULT_DEVICE = "mps"
elif torch.cuda.is_available():
    DEFAULT_DEVICE = "cuda"
else:
    DEFAULT_DEVICE = "cpu"

# Default Generation Parameters
GEN_CONFIG = {
    "max_new_tokens": 256,
    "temperature": 0.1,  # Low temperature for highly deterministic generation & comparative consistency
    "do_sample": False,
}

# Benchmarking Prompts
TEST_PROMPTS = [
    {
        "id": "short_explainer",
        "description": "Short Explainer (Theory)",
        "prompt": "Explain the concept of multi-token prediction (MTP) in large language models in one paragraph."
    },
    {
        "id": "code_generation",
        "description": "Medium Code Generation",
        "prompt": "Write a highly optimized Python function to find all prime numbers up to N using the Sieve of Eratosthenes. Include docstrings and complexity analysis."
    },
    {
        "id": "reasoning_riddle",
        "description": "Logic/Reasoning Puzzle",
        "prompt": "A farmer has 17 sheep, and all but 9 die. How many sheep does the farmer have left? Explain your step-by-step reasoning."
    },
    {
        "id": "long_writing",
        "description": "Long Form Content Writing",
        "prompt": "Draft a comprehensive blog post introduction about the future of on-device AI models running on edge hardware, highlighting how latency and privacy trade-offs are changing."
    }
]
