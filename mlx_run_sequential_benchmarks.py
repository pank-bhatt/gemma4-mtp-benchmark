import os
import sys
import subprocess
import logging
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/mlx_sequential_runs.log", mode="a")
    ]
)
logger = logging.getLogger("MLXSequentialOrchestrator")

from mlx_run_benchmark import is_model_fully_cached

def parse_args():
    parser = argparse.ArgumentParser(description="Gemma 4 Multi-Model MLX Sequential Orchestrator")
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
    logger.info(f"STARTING SEQUENTIAL MLX BENCHMARK PIPELINE FOR: Gemma 4 {size.upper()} ({bits}-bit)")
    logger.info(f"===========================================================")
    
    cmd = ["venv/bin/python", "mlx_run_benchmark.py", "--size", size, "--bits", str(bits)]
    
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
            report_name = f"mlx_results_detailed_{size}_4bit.md" if bits == 4 else f"mlx_results_detailed_{size}.md"
            logger.info(f"✅ SUCCESS: MLX Benchmark for Gemma 4 {size.upper()} completed successfully!")
            logger.info(f"Report written to: docs/{report_name}")
        else:
            logger.error(f"❌ ERROR: MLX Benchmark for Gemma 4 {size.upper()} failed with exit code {process.returncode}")
            
        return process.returncode == 0
        
    except Exception as e:
        logger.error(f"Exception while running MLX benchmark for {size.upper()}: {e}")
        return False

def main():
    args = parse_args()
    bits = args.bits

    # Map size to Hugging Face MLX model IDs to check cache
    model_mappings = {
        "e2b": {
            "target_4bit": "mlx-community/gemma-4-e2b-it-4bit",
            "target_16bit": "mlx-community/gemma-4-e2b-it-bf16",
            "assistant": "mlx-community/gemma-4-E2B-it-assistant-bf16"
        },
        "e4b": {
            "target_4bit": "mlx-community/gemma-4-e4b-it-4bit",
            "target_16bit": "mlx-community/gemma-4-e4b-it-bf16",
            "assistant": "mlx-community/gemma-4-E4B-it-assistant-bf16"
        },
        "26b": {
            "target_4bit": "mlx-community/gemma-4-26b-a4b-it-4bit",
            "target_16bit": "mlx-community/gemma-4-26b-a4b-it-bf16",
            "assistant": "mlx-community/gemma-4-26B-A4B-it-assistant-bf16"
        },
        "31b": {
            "target_4bit": "mlx-community/gemma-4-31b-it-4bit",
            "target_16bit": "mlx-community/gemma-4-31b-it-bf16",
            "assistant": "mlx-community/gemma-4-31B-it-assistant-bf16"
        }
    }

    logger.info(f"Starting Gemma 4 Multi-Model MLX Sequential Benchmarking Suite ({bits}-bit)")
    
    sizes = [s.strip().lower() for s in args.sizes.split(",") if s.strip()]
    logger.info(f"Target Sizes: {' -> '.join([s.upper() for s in sizes])}")
    cached_sizes = []

    logger.info("Verifying local cache availability for sequential MLX benchmark targets...")
    for size in sizes:
        target_id = model_mappings[size]["target_4bit"] if bits == 4 else model_mappings[size]["target_16bit"]
        assistant_id = model_mappings[size]["assistant"]
        
        target_cached, _ = is_model_fully_cached(target_id)
        assistant_cached, _ = is_model_fully_cached(assistant_id)
        
        if target_cached and assistant_cached:
            cached_sizes.append(size)
        else:
            logger.info(f"⏭️ Skipping Gemma 4 {size.upper()} (not fully cached locally in MLX format).")
            logger.info(f"    To run this benchmark, please download it first using: python download_model.py {size} --mlx")
            
    if not cached_sizes:
        logger.error("❌ ERROR: No fully cached MLX models were found. Please download at least one model first using: python download_model.py {size} --mlx")
        sys.exit(1)
        
    logger.info(f"Locally available MLX models to benchmark: {', '.join([s.upper() for s in cached_sizes])}")
    logger.info("===========================================================")
    
    for size in cached_sizes:
        success = run_benchmark(size, bits)
        if not success:
            logger.warning(f"⚠️ Warning: MLX Benchmark for size {size.upper()} did not complete successfully. Moving to next model...")
            
    logger.info("===========================================================")
    logger.info("ALL AVAILABLE SEQUENTIAL MLX BENCHMARKS COMPLETED!")
    logger.info("===========================================================")

if __name__ == "__main__":
    main()
