import os
import sys

# Prepend venv/bin to PATH to ensure virtual environment tools (like ninja) are visible to subprocesses
venv_bin = os.path.abspath(os.path.join(os.path.dirname(__file__), "venv", "bin"))
os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")

import subprocess
import logging


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/sequential_runs.log", mode="a")
    ]
)
logger = logging.getLogger("SequentialOrchestrator")

import argparse
from run_benchmark import is_model_fully_cached

def parse_args():
    parser = argparse.ArgumentParser(description="Gemma 4 Multi-Model Sequential Orchestrator")
    parser.add_argument(
        "--bits",
        type=int,
        default=16,
        choices=[16, 4],
        help="Quantization level in bits (16 or 4)"
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default="e2b,e4b,26b,31b",
        help="Comma-separated list of model sizes to benchmark (e.g., e2b,e4b)"
    )
    return parser.parse_args()

def run_benchmark(size, bits):
    logger.info(f"===========================================================")
    logger.info(f"STARTING SEQUENTIAL BENCHMARK PIPELINE FOR: Gemma 4 {size.upper()} ({bits}-bit)")
    logger.info(f"===========================================================")
    
    cmd = ["venv/bin/python", "run_benchmark.py", "--size", size, "--bits", str(bits)]
    
    try:
        # Run process and stream logs in real time
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            
        process.wait()
        
        if process.returncode == 0:
            report_name = f"pytorch_results_detailed_{size}_4bit.md" if bits == 4 else f"pytorch_results_detailed_{size}.md"

            logger.info(f"✅ SUCCESS: Benchmark for Gemma 4 {size.upper()} completed successfully!")
            logger.info(f"Report written to: docs/{report_name}")
        else:
            logger.error(f"❌ ERROR: Benchmark for Gemma 4 {size.upper()} failed with exit code {process.returncode}")
            
        return process.returncode == 0
        
    except Exception as e:
        logger.error(f"Exception while running benchmark for {size.upper()}: {e}")
        return False

def main():
    args = parse_args()
    bits = args.bits

    # Map size to Hugging Face model IDs to check cache
    model_mappings = {
        "e2b": {
            "target": "google/gemma-4-E2B-it",
            "assistant": "google/gemma-4-E2B-it-assistant"
        },
        "e4b": {
            "target": "google/gemma-4-E4B-it",
            "assistant": "google/gemma-4-E4B-it-assistant"
        },
        "26b": {
            "target": "google/gemma-4-26B-A4B-it",
            "assistant": "google/gemma-4-26B-A4B-it-assistant"
        },
        "31b": {
            "target": "google/gemma-4-31B-it",
            "assistant": "google/gemma-4-31B-it-assistant"
        }
    }

    logger.info(f"Starting Gemma 4 Multi-Model Sequential Benchmarking Suite ({bits}-bit)")
    
    sizes = [s.strip().lower() for s in args.sizes.split(",") if s.strip()]
    logger.info(f"Target Sizes: {' -> '.join([s.upper() for s in sizes])}")
    cached_sizes = []

    logger.info("Verifying local cache availability for sequential benchmark targets...")
    for size in sizes:
        target_id = model_mappings[size]["target"]
        assistant_id = model_mappings[size]["assistant"]
        
        target_cached, _ = is_model_fully_cached(target_id)
        assistant_cached, _ = is_model_fully_cached(assistant_id)
        
        if target_cached and assistant_cached:
            cached_sizes.append(size)
        else:
            logger.info(f"⏭️ Skipping Gemma 4 {size.upper()} (not fully cached locally).")
            logger.info(f"    To run this benchmark, please download it first using: python download_model.py --size {size}")
            
    if not cached_sizes:
        logger.error("❌ ERROR: No fully cached models were found. Please download at least one model first using: python download_model.py --size {size}")
        sys.exit(1)
        
    logger.info(f"Locally available models to benchmark: {', '.join([s.upper() for s in cached_sizes])}")
    logger.info("===========================================================")
    
    for size in cached_sizes:
        success = run_benchmark(size, bits)
        if not success:
            logger.warning(f"⚠️ Warning: Benchmark for size {size.upper()} did not complete successfully. Moving to next model...")
            
    logger.info("===========================================================")
    logger.info("ALL AVAILABLE SEQUENTIAL BENCHMARKS COMPLETED!")
    logger.info("===========================================================")

if __name__ == "__main__":
    main()

