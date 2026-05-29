import os
import sys
import time
import argparse
import logging
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tabulate import tabulate

import config
from profiler import ResourceProfiler

# Setup logging
os.makedirs(config.LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.EXECUTION_LOG_PATH, mode="a")
    ]
)
logger = logging.getLogger("Gemma4MTPBenchmark")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Gemma 4 E2B MTP Inferencing Benchmark Suite")
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Use lightweight dummy models (facebook/opt-125m) to verify script plumbing without HF gated model access."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config.DEFAULT_DEVICE,
        help=f"Device to run inference on (default: {config.DEFAULT_DEVICE})"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.environ.get("HF_TOKEN", ""),
        help="Hugging Face User Access Token (optional if logged in via CLI)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=config.GEN_CONFIG["max_new_tokens"],
        help=f"Max new tokens to generate (default: {config.GEN_CONFIG['max_new_tokens']})"
    )
    return parser.parse_args()


def check_hf_access(model_id, token):
    """Inform the user and check Hugging Face hub authentication."""
    if "gemma" in model_id.lower() and not token:
        # Check if already authenticated via caching or cli
        try:
            from huggingface_hub import HfFolder
            cached_token = HfFolder.get_token()
            if cached_token:
                logger.info("Found cached Hugging Face token in environment/cli cache.")
                return True
        except ImportError:
            pass
        
        logger.warning(
            f"You are attempting to load '{model_id}' which is a gated repository. "
            "If you experience loading errors, please make sure you have accepted the model license "
            "on Hugging Face and provided your token via --hf-token or the HF_TOKEN environment variable."
        )
    return True


def patch_tokenizer_config(model_id, token=None):
    """
    Downloads the tokenizer_config.json for a given model ID and patches
    the `extra_special_tokens` key if it is saved as a list, which causes
    an AttributeError: 'list' object has no attribute 'keys' in some
    transformers versions.
    """
    if "gemma" not in model_id.lower():
        return
        
    try:
        from pathlib import Path
        from huggingface_hub import hf_hub_download
        import json
        
        logger.info(f"Checking tokenizer config for '{model_id}'...")
        config_path = Path(hf_hub_download(
            repo_id=model_id,
            filename="tokenizer_config.json",
            token=token if token else None
        ))
        
        if config_path.exists():
            with open(config_path, "r") as f:
                config_data = json.load(f)
                
            if isinstance(config_data.get("extra_special_tokens"), list):
                logger.info(f"Detected list in 'extra_special_tokens' for {model_id}. Patching to empty dictionary...")
                config_data["extra_special_tokens"] = {}
                with open(config_path, "w") as f:
                    json.dump(config_data, f, indent=2)
                logger.info(f"Successfully patched tokenizer config for '{model_id}'!")
            else:
                logger.info(f"Tokenizer config for '{model_id}' is already in correct format.")
    except Exception as e:
        logger.warning(f"Skipped tokenizer config patching for '{model_id}' due to: {e}")


def load_tokenizer_and_models(args):
    """Loads target and assistant tokenizer and models onto the designated device."""
    if args.dummy:
        target_id = config.DUMMY_TARGET_MODEL
        assistant_id = config.DUMMY_ASSISTANT_MODEL
        logger.info(f"Running in [DUMMY MODE] using lightweight models:")
        logger.info(f"  Target:    {target_id}")
        logger.info(f"  Assistant: {assistant_id}")
    else:
        target_id = config.TARGET_MODEL
        assistant_id = config.ASSISTANT_MODEL
        logger.info(f"Running in [PRODUCTION MODE] using Gemma 4:")
        logger.info(f"  Target:    {target_id}")
        logger.info(f"  Assistant: {assistant_id}")
        
    check_hf_access(target_id, args.hf_token)

    # Patch tokenizer configs to bypass list-keys mismatch bug
    if not args.dummy:
        patch_tokenizer_config(target_id, args.hf_token)
        patch_tokenizer_config(assistant_id, args.hf_token)

    # Determine optimal precision (float16 is standard and highly optimized on MPS / CUDA)
    device = args.device.lower()
    if device in ["mps", "cuda"]:
        torch_dtype = torch.float16
        logger.info(f"Selected device '{device}'. Loading models in FP16 precision.")
    else:
        torch_dtype = torch.float32
        logger.info(f"Selected device '{device}'. Loading models in FP32 precision.")

    # 1. Load Tokenizer
    logger.info(f"Loading tokenizer for target '{target_id}'...")
    tokenizer_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        target_id,
        token=args.hf_token if args.hf_token else None,
        trust_remote_code=True
    )
    # OPT tokenizer doesn't have a default pad token, pad to eos_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    logger.info(f"Tokenizer loaded in {time.time() - tokenizer_start:.2f} seconds.")

    # 2. Load Target Model
    logger.info(f"Loading target model '{target_id}' onto {device}...")
    model_start = time.time()
    target_model = AutoModelForCausalLM.from_pretrained(
        target_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        token=args.hf_token if args.hf_token else None,
        trust_remote_code=True
    ).to(device)
    target_model.eval()  # Set evaluation mode
    logger.info(f"Target model loaded in {time.time() - model_start:.2f} seconds.")

    # 3. Load Assistant Model for MTP
    logger.info(f"Loading assistant model '{assistant_id}' onto {device} for MTP...")
    assistant_start = time.time()
    assistant_model = AutoModelForCausalLM.from_pretrained(
        assistant_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        token=args.hf_token if args.hf_token else None,
        trust_remote_code=True
    ).to(device)
    assistant_model.eval()  # Set evaluation mode
    logger.info(f"Assistant model loaded in {time.time() - assistant_start:.2f} seconds.")

    return tokenizer, target_model, assistant_model


def run_single_inference(tokenizer, model, prompt_text, device, max_tokens, assistant_model=None):
    """Executes a single model generation step under resource profiling."""
    # Tokenize input
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    # Initialize configuration
    gen_args = {
        "max_new_tokens": max_tokens,
        "temperature": config.GEN_CONFIG["temperature"],
        "do_sample": config.GEN_CONFIG["do_sample"],
        "pad_token_id": tokenizer.pad_token_id,
    }

    if assistant_model is not None:
        gen_args["assistant_model"] = assistant_model

    # Run warm generation to capture Time to First Token (TTFT)
    # We do a tiny generation of 1 token to measure prompt encoding latency / TTFT
    ttft_args = gen_args.copy()
    ttft_args["max_new_tokens"] = 1
    
    start_ttft = time.time()
    with torch.no_grad():
        _ = model.generate(**inputs, **ttft_args)
    ttft = (time.time() - start_ttft) * 1000.0  # in ms

    # Now execute complete generation with profiling
    profiler_label = "mtp" if assistant_model is not None else "baseline"
    prof_name = f"{profiler_label}_{int(time.time())}"
    prof = ResourceProfiler(interval_sec=0.05, profile_name=prof_name)
    
    prof.start()
    start_gen = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_args)
    end_gen = time.time()
    prof_summary = prof.stop(save_dir=config.RAW_PROFILES_DIR)

    # Decode and compute stats
    total_tokens = len(outputs[0])
    new_tokens = total_tokens - input_len
    generation_time = end_gen - start_gen
    
    # Calculate speed
    tokens_per_sec = new_tokens / generation_time if generation_time > 0 else 0.0
    output_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

    metrics = {
        "mode": "MTP" if assistant_model is not None else "Baseline",
        "input_tokens": input_len,
        "output_tokens": new_tokens,
        "generation_time_sec": round(generation_time, 3),
        "tokens_per_sec": round(tokens_per_sec, 2),
        "ttft_ms": round(ttft, 2),
        "avg_cpu_percent": prof_summary.get("avg_cpu_percent", 0.0),
        "peak_cpu_percent": prof_summary.get("peak_cpu_percent", 0.0),
        "start_memory_mb": prof_summary.get("start_memory_mb", 0.0),
        "peak_memory_mb": prof_summary.get("peak_memory_mb", 0.0),
        "memory_growth_mb": prof_summary.get("memory_growth_mb", 0.0),
        "raw_text_length": len(output_text)
    }

    return metrics, output_text


def execute_benchmarks(tokenizer, target_model, assistant_model, args):
    """Runs the benchmark pipeline across baseline and MTP configurations for all test prompts."""
    results = []
    
    # Warm-up phase
    logger.info("=========================================")
    logger.info("Executing model warm-ups to compile GPU kernels...")
    warmup_prompt = "Warm-up inference check."
    _, _ = run_single_inference(
        tokenizer, target_model, warmup_prompt, args.device, max_tokens=10
    )
    _, _ = run_single_inference(
        tokenizer, target_model, warmup_prompt, args.device, max_tokens=10, assistant_model=assistant_model
    )
    logger.info("Warm-up complete.")
    logger.info("=========================================\n")

    for test_idx, item in enumerate(config.TEST_PROMPTS):
        p_id = item["id"]
        p_desc = item["description"]
        prompt = item["prompt"]
        
        logger.info(f"[{test_idx + 1}/{len(config.TEST_PROMPTS)}] Running Test: {p_desc} ({p_id})")
        logger.info(f"Prompt: \"{prompt[:60]}...\"")

        # 1. Run Baseline
        logger.info("  Executing Baseline Generation...")
        base_metrics, _ = run_single_inference(
            tokenizer, target_model, prompt, args.device, max_tokens=args.max_tokens
        )
        base_metrics["prompt_id"] = p_id
        base_metrics["prompt_desc"] = p_desc
        logger.info(
            f"    Baseline Result: {base_metrics['tokens_per_sec']} t/s, "
            f"Peak RAM: {base_metrics['peak_memory_mb']:.1f} MB, "
            f"Avg CPU: {base_metrics['avg_cpu_percent']}%"
        )

        # 2. Run MTP
        logger.info("  Executing MTP Speculative Generation...")
        mtp_metrics, _ = run_single_inference(
            tokenizer, target_model, prompt, args.device, max_tokens=args.max_tokens, assistant_model=assistant_model
        )
        mtp_metrics["prompt_id"] = p_id
        mtp_metrics["prompt_desc"] = p_desc
        logger.info(
            f"    MTP Result:      {mtp_metrics['tokens_per_sec']} t/s, "
            f"Peak RAM: {mtp_metrics['peak_memory_mb']:.1f} MB, "
            f"Avg CPU: {mtp_metrics['avg_cpu_percent']}%"
        )

        # Calculate comparison speedup and overheads
        speedup = mtp_metrics["tokens_per_sec"] / base_metrics["tokens_per_sec"] if base_metrics["tokens_per_sec"] > 0 else 1.0
        mem_increase = mtp_metrics["peak_memory_mb"] - base_metrics["peak_memory_mb"]
        cpu_diff = mtp_metrics["avg_cpu_percent"] - base_metrics["avg_cpu_percent"]

        logger.info(
            f"    Speedup: {speedup:.2f}x | "
            f"RAM delta: {mem_increase:+.1f} MB | "
            f"CPU delta: {cpu_diff:+.1f}%"
        )
        logger.info("-" * 50)

        # Record metrics
        results.append(base_metrics)
        results.append(mtp_metrics)

    # 3. Log results to CSV
    logger.info("Aggregating results and updating CSV database...")
    df_results = pd.DataFrame(results)
    
    # Save/Append to CSV
    if os.path.exists(config.CSV_RESULTS_PATH):
        try:
            df_old = pd.read_csv(config.CSV_RESULTS_PATH)
            df_new = pd.concat([df_old, df_results], ignore_index=True)
            df_new.to_csv(config.CSV_RESULTS_PATH, index=False)
        except Exception:
            df_results.to_csv(config.CSV_RESULTS_PATH, index=False)
    else:
        df_results.to_csv(config.CSV_RESULTS_PATH, index=False)

    logger.info(f"Results recorded in: {config.CSV_RESULTS_PATH}")
    
    return results


def print_comparison_report(results, is_dummy):
    """Builds and prints a high-fidelity visual summary table comparing Baseline vs MTP."""
    # Organize data into a pivot-friendly structure
    paired_runs = {}
    for r in results:
        p_id = r["prompt_id"]
        p_desc = r["prompt_desc"]
        mode = r["mode"]
        if p_id not in paired_runs:
            paired_runs[p_id] = {"desc": p_desc}
        paired_runs[p_id][mode] = r

    headers = [
        "Benchmark Prompt", "Mode", "Speed (t/s)", "Speedup", "Peak Memory (MB)", "Mem Delta", "Avg CPU (%)", "CPU Delta"
    ]
    
    table_data = []
    
    overall_base_speed = 0.0
    overall_mtp_speed = 0.0
    count = 0
    
    for p_id, modes in paired_runs.items():
        base = modes.get("Baseline")
        mtp = modes.get("MTP")
        if not base or not mtp:
            continue
            
        speedup = mtp["tokens_per_sec"] / base["tokens_per_sec"] if base["tokens_per_sec"] > 0 else 1.0
        mem_delta = mtp["peak_memory_mb"] - base["peak_memory_mb"]
        cpu_delta = mtp["avg_cpu_percent"] - base["avg_cpu_percent"]
        
        overall_base_speed += base["tokens_per_sec"]
        overall_mtp_speed += mtp["tokens_per_sec"]
        count += 1
        
        # Add baseline row
        table_data.append([
            modes["desc"],
            "Baseline",
            f"{base['tokens_per_sec']:.2f}",
            "1.00x (Ref)",
            f"{base['peak_memory_mb']:.1f}",
            "-",
            f"{base['avg_cpu_percent']:.1f}%",
            "-"
        ])
        
        # Add MTP row
        table_data.append([
            "",
            "MTP (Spec)",
            f"{mtp['tokens_per_sec']:.2f}",
            f"{speedup:.2f}x",
            f"{mtp['peak_memory_mb']:.1f}",
            f"{mem_delta:+.1f} MB",
            f"{mtp['avg_cpu_percent']:.1f}%",
            f"{cpu_delta:+.1f}%"
        ])
        
        # Divider row
        table_data.append(["-"*30, "-"*10, "-"*10, "-"*10, "-"*15, "-"*10, "-"*12, "-"*10])

    # Remove the last divider
    if table_data:
        table_data.pop()

    # Calculate average performance gains
    avg_speedup = (overall_mtp_speed / overall_base_speed) if overall_base_speed > 0 else 1.0
    
    title = f"\n🔥 GEMMA 4 E2B MTP EXPERIMENT PERFORMANCE REPORT {'[DUMMY MODEL RUN]' if is_dummy else ''} 🔥"
    logger.info(title)
    logger.info("=" * len(title))
    logger.info("\n" + tabulate(table_data, headers=headers, tablefmt="fancy_grid"))
    
    logger.info("\n🏁 OVERALL SUMMARY:")
    logger.info(f"  • Average Baseline Speed:  {overall_base_speed/count if count else 0:.2f} tokens/second")
    logger.info(f"  • Average MTP Speed:       {overall_mtp_speed/count if count else 0:.2f} tokens/second")
    logger.info(f"  • Average MTP Performance Boost:  \033[1;32m{avg_speedup:.2f}x faster\033[0m")
    logger.info(f"  • Raw metrics logged to:    {config.CSV_RESULTS_PATH}")
    logger.info(f"  • Background system trace: {config.RAW_PROFILES_DIR}\n")


def main():
    args = parse_arguments()
    logger.info("=========================================")
    logger.info("Initializing Gemma 4 MTP Benchmarking Tool")
    logger.info(f"System Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Host Device: {args.device.upper()}")
    logger.info("=========================================")
    
    try:
        # 1. Load tokenizer and models
        tokenizer, target_model, assistant_model = load_tokenizer_and_models(args)
        
        # 2. Run benchmarks
        results = execute_benchmarks(tokenizer, target_model, assistant_model, args)
        
        # 3. Print final formatted table
        print_comparison_report(results, args.dummy)
        
    except KeyboardInterrupt:
        logger.info("\nBenchmark cancelled by user.")
    except Exception as e:
        logger.exception(f"Fatal error during benchmark execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
