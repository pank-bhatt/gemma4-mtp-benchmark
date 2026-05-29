import os
import sys
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

def parse_args():
    parser = argparse.ArgumentParser(description="Gemma 4 Multi-Model Sequential Orchestrator")
    parser.add_argument(
        "--bits",
        type=int,
        default=16,
        choices=[16, 4],
        help="Quantization level in bits (16 or 4)"
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
            report_name = f"results_detailed_{size}_4bit.md" if bits == 4 else f"results_detailed_{size}.md"
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

    logger.info(f"Starting Gemma 4 Multi-Model Sequential Benchmarking Suite ({bits}-bit)")
    logger.info("Target Sizes: E2B -> E4B -> 26B -> 31B")
    logger.info("Download progress will be automatically tracked in logs/models.log")
    
    sizes = ["e2b", "e4b", "26b", "31b"]
    
    for size in sizes:
        success = run_benchmark(size, bits)
        if not success:
            logger.warning(f"⚠️ Warning: Benchmark for size {size.upper()} did not complete successfully. Moving to next model...")
            
    logger.info("===========================================================")
    logger.info("ALL BENCHMARKS COMPLETED!")
    logger.info("===========================================================")

if __name__ == "__main__":
    main()
