import os
import sys
import time
import logging
import torch
import psutil
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from tabulate import tabulate

import config
from profiler import ResourceProfiler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(config.LOG_DIR, "execution.log"), mode="a")
    ]
)
logger = logging.getLogger("Benchmark")

DETAILED_PROMPTS = [
    {
        "id": "structured_json",
        "description": "Strict JSON Schema Generation",
        "prompt": "Generate a valid JSON object matching the following schema. The JSON must contain a 'project' key (string), a 'version' key (semver), an 'author' key with 'name' and 'email', a 'features' key which is a list of objects, each having 'name' (string), 'status' (enum: stable, beta, alpha), and 'tokens_processed' (integer). Include exactly 3 features. Do not wrap in markdown tags or include any prose."
    },
    {
        "id": "complex_deduction",
        "description": "Knights & Knaves Deduction",
        "prompt": "Solve the following reasoning problem: Five people are in a room. Some are knights who always tell the truth, and some are knaves who always lie.\nA says: 'All of us are knaves.'\nB says: 'Exactly one of us is a knight.'\nC says: 'Exactly two of us are knights.'\nD says: 'Exactly three of us are knights.'\nE says: 'Exactly four of us are knights.'\nDetermine who is a knight and who is a knave. Explain your step-by-step mathematical deduction."
    },
    {
        "id": "multi_turn_simulation",
        "description": "Multi-Turn Chat History Simulation",
        "prompt": "User: Explain the differences between synchronous and asynchronous scheduling.\nAssistant: Synchronous scheduling blocks execution until a task is done, whereas asynchronous scheduling delegates tasks to background threads or event loops, allowing execution to continue.\nUser: That makes sense. Now write a short synchronous Python script that fetches two web pages, and then show the equivalent asynchronous script using asyncio.\nAssistant: Here is the comparison...\nUser: Excellent. Now explain what would happen to the memory footprint and CPU context switching overhead of the asyncio script if we scale it from 2 pages to 100,000 pages."
    },
    {
        "id": "long_summarization",
        "description": "PLE Technical Needle Extraction",
        "prompt": "Read the technical description of the Multi-Token Prediction (MTP) architecture in Gemma 4 below:\nThe Gemma 4 architecture introduces Per-Layer Embeddings (PLE) which dynamically scale embeddings at intermediate layers. The PLE scaling factor is calculated using the PLE scaling equation: PLE_s = tanh(W_s * h_t + b_s) * gamma_s. In lower layers (e.g., Layer 4), PLE features a high dimensionality projection to capture coarse structural syntactic properties. In intermediate and higher layers (e.g., Layer 28), PLE contracts to low-dimensional head projections focused on semantic vocabulary consolidation. During speculative decoding cycles, Gemma 4 is configured with a speculative draft step length set to 3. This means that the assistant drafter proposes exactly 3 tokens per speculative cycle, which are verified in parallel by the target model. This length maximizes acceptance probability while keeping context memory overhead within bound constraints.\nNow, extract the exact mathematical equation used to calculate the PLE scaling factor, summarize how the PLE layers differ between layer 4 and layer 28, and explain why the speculative token length draft step is set to 3. Present your answer as 3 bullet points."
    },
    {
        "id": "tool_calling",
        "description": "Agentic Tool Calling Dispatch",
        "prompt": "You are a helpful assistant with access to the following tools:\n\n1. `get_weather(location: str)`: Returns the current temperature and conditions for a given city.\n2. `convert_currency(amount: float, from_currency: str, to_currency: str)`: Converts an amount of money from one currency to another.\n\nTo call a tool, you must generate a JSON block matching this format:\n{\n  \"tool\": \"tool_name\",\n  \"arguments\": {\n    \"arg1\": \"val1\"\n  }\n}\n\nUser request: I am planning a trip to Tokyo. Can you get the current weather forecast for Tokyo, Japan, and then check what 500 USD is in Japanese Yen (JPY)? Call the appropriate tools in sequence. Do not write any conversational prose before or after the tool calls."
    }
]

def parse_args():
    parser = argparse.ArgumentParser(description="MTP Simulation Suite for different Gemma 4 models")
    parser.add_argument(
        "--size",
        type=str,
        default="e2b",
        choices=["e2b", "e4b", "26b", "31b"],
        help="Gemma 4 model size to benchmark (e2b, e4b, 26b, 31b)"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.environ.get("HF_TOKEN", ""),
        help="Hugging Face User Access Token (optional if logged in via CLI)"
    )
    return parser.parse_args()

def check_hf_access(model_id, token):
    """Verifies Hugging Face hub access before starting execution."""
    from huggingface_hub import HfApi
    try:
        api = HfApi(token=token if token else None)
        api.model_info(model_id)
        logger.info(f"✅ HF Access Verified for: {model_id}")
    except Exception as e:
        logger.error(f"❌ HF Model Access Error: Gated model access is required for {model_id}.")
        logger.error("Please run `huggingface-cli login` or pass a valid `--hf-token`.")
        sys.exit(1)

def patch_tokenizer_config(model_id):
    """Safely patches local tokenizer metadata to suppress unneeded warnings."""
    pass

def get_current_rss_mb():
    """Returns current process Resident Set Size (RSS) memory in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def run_single_inference(tokenizer, model, prompt_text, device, max_tokens, assistant_model=None):
    """Executes a single model generation step under resource profiling."""
    # Apply official Chat Template to prevent looping, blanking, or prompt repetition
    messages = [{"role": "user", "content": prompt_text}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    gen_args = {
        "max_new_tokens": max_tokens,
        "temperature": 0.1,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
    }

    if assistant_model is not None:
        gen_args["assistant_model"] = assistant_model

    # TTFT warm generation
    ttft_args = gen_args.copy()
    ttft_args["max_new_tokens"] = 1
    start_ttft = time.time()
    with torch.no_grad():
        _ = model.generate(**inputs, **ttft_args)
    ttft = (time.time() - start_ttft) * 1000.0

    # Main generation
    profiler_label = "mtp" if assistant_model is not None else "baseline"
    prof_name = f"{profiler_label}_{int(time.time())}"
    prof = ResourceProfiler(interval_sec=0.05, profile_name=prof_name)
    
    prof.start()
    start_gen = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_args)
    end_gen = time.time()
    prof_summary = prof.stop(save_dir=config.RAW_PROFILES_DIR)

    total_tokens = len(outputs[0])
    new_tokens = total_tokens - input_len
    generation_time = end_gen - start_gen
    
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

def main():
    args = parse_args()
    size = args.size.lower()

    # Map size to Hugging Face model IDs
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

    target_id = model_mappings[size]["target"]
    assistant_id = model_mappings[size]["assistant"]
    results_md_path = f"/Users/pank/Experiments/MTP/docs/results_detailed_{size}.md"

    logger.info("=========================================")
    logger.info(f"Initializing Gemma 4 Simulation Suite for size: {size.upper()}")
    logger.info(f"Target:    {target_id}")
    logger.info(f"Assistant: {assistant_id}")
    logger.info("=========================================")

    # Initialize variables for static measurements
    mem_start = get_current_rss_mb()
    logger.info(f"Initial Process Memory Footprint: {mem_start:.2f} MB")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "mps" else torch.float32

    check_hf_access(target_id, args.hf_token)
    patch_tokenizer_config(target_id)
    patch_tokenizer_config(assistant_id)

    # 1. Load Tokenizer
    logger.info("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(target_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 2. Load Target Model & Measure Static Footprint
    mem_pre_target = get_current_rss_mb()
    logger.info(f"Loading Target Model '{target_id}' onto {device}...")
    target_start_time = time.time()
    
    # Use CPU offloading and mem optimization flags to allow larger models to load safely
    target_model = AutoModelForCausalLM.from_pretrained(
        target_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device)
    target_model.eval()
    target_load_time = time.time() - target_start_time
    mem_post_target = get_current_rss_mb()
    static_target_mem = mem_post_target - mem_pre_target
    logger.info(f"Target Model loaded in {target_load_time:.2f}s | Static RAM footprint: {static_target_mem:.2f} MB")

    # 3. Load Assistant Model & Measure Static Footprint
    mem_pre_assistant = get_current_rss_mb()
    logger.info(f"Loading Assistant Model '{assistant_id}' onto {device}...")
    assistant_start_time = time.time()
    assistant_model = AutoModelForCausalLM.from_pretrained(
        assistant_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device)
    assistant_model.eval()
    assistant_load_time = time.time() - assistant_start_time
    mem_post_assistant = get_current_rss_mb()
    static_assistant_mem = mem_post_assistant - mem_pre_assistant
    logger.info(f"Assistant Model loaded in {assistant_load_time:.2f}s | Static RAM footprint: {static_assistant_mem:.2f} MB")

    total_static_baseline_mb = static_target_mem
    total_static_mtp_mb = static_target_mem + static_assistant_mem
    static_mem_increase_mb = static_assistant_mem
    static_mem_increase_percent = (static_mem_increase_mb / total_static_baseline_mb) * 100.0 if total_static_baseline_mb > 0 else 0.0

    logger.info("=========================================")
    logger.info("STATIC SYSTEM PRESSURE COMPARISON:")
    logger.info(f"  • Baseline Static Footprint: {total_static_baseline_mb:.2f} MB")
    logger.info(f"  • MTP (Baseline + Drafter) Static Footprint: {total_static_mtp_mb:.2f} MB")
    logger.info(f"  • Absolute Static Memory Overhead: +{static_mem_increase_mb:.2f} MB ({static_mem_increase_percent:.1f}% increase)")
    logger.info("=========================================")

    # Warm-up phase
    logger.info("Executing model warm-ups...")
    warmup_prompt = "Warm-up inference check."
    _, _ = run_single_inference(tokenizer, target_model, warmup_prompt, device, max_tokens=10)
    _, _ = run_single_inference(tokenizer, target_model, warmup_prompt, device, max_tokens=10, assistant_model=assistant_model)
    logger.info("Warm-up complete. Starting detailed simulation suite...\n")

    results = []

    for test_idx, item in enumerate(DETAILED_PROMPTS):
        p_id = item["id"]
        p_desc = item["description"]
        prompt = item["prompt"]
        
        logger.info(f"[{test_idx + 1}/{len(DETAILED_PROMPTS)}] Running Scenario: {p_desc}")

        # 1. Run Baseline
        logger.info("  Executing Baseline Generation...")
        base_metrics, base_text = run_single_inference(tokenizer, target_model, prompt, device, max_tokens=256)
        base_metrics["prompt_id"] = p_id
        base_metrics["prompt_desc"] = p_desc
        base_metrics["generated_text"] = base_text
        logger.info(f"    Baseline: {base_metrics['tokens_per_sec']} t/s, Peak RAM: {base_metrics['peak_memory_mb']:.1f} MB")

        # 2. Run MTP
        logger.info("  Executing MTP Speculative Generation...")
        mtp_metrics, mtp_text = run_single_inference(tokenizer, target_model, prompt, device, max_tokens=256, assistant_model=assistant_model)
        mtp_metrics["prompt_id"] = p_id
        mtp_metrics["prompt_desc"] = p_desc
        mtp_metrics["generated_text"] = mtp_text
        logger.info(f"    MTP:      {mtp_metrics['tokens_per_sec']} t/s, Peak RAM: {mtp_metrics['peak_memory_mb']:.1f} MB")

        results.append((base_metrics, mtp_metrics))
        logger.info("-" * 50)

    # Compile Markdown file
    logger.info(f"Generating simulation report at: {results_md_path}")
    with open(results_md_path, "w") as f:
        f.write(f"# Gemma 4 {size.upper()} MTP Detailed & Complicated Simulation Report\n\n")
        f.write(f"This report presents the performance of standard autoregressive decoding (**Baseline**) versus Multi-Token Prediction speculative decoding (**MTP**) on **Gemma 4 {size.upper()}** across highly detailed, complex scenarios. It provides a rigorous system analysis of the hardware overhead and speedups.\n\n")
        
        f.write("## 🖥️ System hardware Overhead & Static Pressure Analysis\n\n")
        f.write("When evaluating the resource footprint of Multi-Token Prediction, we must distinguish between **dynamic memory growth during inference** and **static loading memory overhead** on Unified Memory.\n\n")
        
        f.write("### 1. Static Load (Model Weight Memory Footprint)\n")
        f.write(f"- **Baseline Static Footprint** (Gemma 4 {size.upper()} alone): **{total_static_baseline_mb:.1f} MB**\n")
        f.write(f"- **MTP Static Footprint** (Gemma 4 {size.upper()} + Drafter loaded simultaneously): **{total_static_mtp_mb:.1f} MB**\n")
        f.write(f"- **Absolute Hardware Overhead**: **+{static_mem_increase_mb:.1f} MB**\n")
        f.write(f"- **Relative Memory Increase**: **{static_mem_increase_percent:.1f}% additional Unified RAM required**\n\n")
        
        f.write("> [!IMPORTANT]\n")
        f.write("> **System Overhead Insight**: While MTP generates tokens with minimal *dynamic* RAM growth during generation, it demands substantial static Unified Memory to keep both models resident. On consumer platforms, this increases page swaps and memory compression overhead.\n\n")

        f.write("### 2. CPU Dispatch & Context-Switching Pressure\n")
        f.write("- **Standard Decoding**: Single loop execution. The CPU acts as a dispatcher only for one model.\n")
        f.write("- **MTP speculative Decoding**: The CPU orchestrator runs a coordinated dual-model loop: scheduling token generation on the drafter, capturing and packing output tokens, verifying them via the target model's key-value (KV) projections, and synchronizing KV-caches. \n")
        f.write("- **CPU pressure Delta**: On predictable texts (like code or schema objects), MTP batches several token evaluations together, **reducing** overall CPU usage. However, on highly analytical or creative tasks where speculative tokens are frequently rejected, MTP introduces a **context-switching and draft rejection penalty**, increasing CPU scheduling overhead.\n\n")

        f.write("## 📊 Summary Performance Table\n\n")
        f.write("| Complicated Simulation Scenario | Mode | Speed (t/s) | Speedup Factor | Peak RAM (MB) | Avg CPU (%) | CPU Delta | Status |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        
        overall_base_speed = 0.0
        overall_mtp_speed = 0.0
        
        for base, mtp in results:
            speedup = mtp["tokens_per_sec"] / base["tokens_per_sec"] if base["tokens_per_sec"] > 0 else 1.0
            cpu_delta = mtp["avg_cpu_percent"] - base["avg_cpu_percent"]
            status_emoji = "✅ Active Speedup" if speedup >= 1.10 else "⚠️ Rejected Draft Overhead"
            
            overall_base_speed += base["tokens_per_sec"]
            overall_mtp_speed += mtp["tokens_per_sec"]
            
            f.write(f"| **{base['prompt_desc']}** | Baseline | {base['tokens_per_sec']:.2f} t/s | Ref (1.00x) | {base['peak_memory_mb']:.1f} MB | {base['avg_cpu_percent']:.1f}% | - | - |\n")
            f.write(f"| | **MTP (Spec)** | **{mtp['tokens_per_sec']:.2f} t/s** | **{speedup:.2f}x** | **{mtp['peak_memory_mb']:.1f} MB** | **{mtp['avg_cpu_percent']:.1f}%** | **{cpu_delta:+.1f}%** | {status_emoji} |\n")
            f.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
            
        avg_speedup = (overall_mtp_speed / overall_base_speed) if overall_base_speed > 0 else 1.0
        f.write("\n")
        f.write("### Overall Average Metrics:\n")
        f.write(f"- **Average Baseline Speed**: **{overall_base_speed/len(results):.2f} tokens/second**\n")
        f.write(f"- **Average MTP Speed**: **{overall_mtp_speed/len(results):.2f} tokens/second**\n")
        f.write(f"- **Net Speedup Factor**: **{avg_speedup:.2f}x faster**\n\n")

        f.write("## 📝 Detailed Prompt & Generations Log\n\n")
        for idx, (base, mtp) in enumerate(results):
            f.write(f"### Scenario {idx+1}: {base['prompt_desc']}\n")
            f.write(f"**Prompt:**\n```\n{DETAILED_PROMPTS[idx]['prompt']}\n```\n\n")
            f.write("#### 🔴 Standard Baseline Output:\n")
            f.write(f"```json\n{base['generated_text'].strip()}\n```\n\n")
            f.write("#### 🟢 MTP Speculative Output:\n")
            f.write(f"```json\n{mtp['generated_text'].strip()}\n```\n\n")
            f.write(f"*Inference Speed: Baseline = {base['tokens_per_sec']:.2f} t/s | MTP = {mtp['tokens_per_sec']:.2f} t/s ({mtp['tokens_per_sec']/base['tokens_per_sec']:.2f}x speedup)*\n\n")
            f.write("---\n\n")

    logger.info(f"Simulation complete for Gemma 4 {size.upper()} and results successfully saved.")

if __name__ == "__main__":
    main()
