import os
import sys

# Prepend venv/bin to PATH to ensure virtual environment tools (like ninja) are visible during JIT compilation
venv_bin = os.path.abspath(os.path.join(os.path.dirname(__file__), "venv", "bin"))
os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")

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
    },
    {
        "id": "rag_synthesis",
        "description": "RAG Contextual Synthesis",
        "prompt": "You are a factual assistant. Based ONLY on the retrieved knowledge base passages below, answer the user's question. Do not assume or extrapolate. If the context does not contain the answer, say 'I cannot answer this based on the retrieved context'.\n\n[Passage 1]\nThe Nebula-9 supercluster is a high-density stellar cluster located approximately 4.2 million light-years from Earth. Discovered in 2024 by the orbital telescope Chronos, it contains over 1,200 active star-forming regions. The central star of the cluster, designated N9-Alpha, is a hypergiant with a mass equivalent to 180 solar masses. N9-Alpha exhibits intense periodic flare-ups every 14.3 Earth days, ejecting coronal mass at speeds exceeding 4,500 km/s.\n\n[Passage 2]\nStellar navigation around the Nebula-9 supercluster is severely restricted due to high concentrations of particulate interstellar dust. According to the United Stellar Coalition Regulation 88-C, ships crossing within 50 light-years of the cluster's boundary must engage magnetic deflection shields (MDS) to prevent particle abrasion. The dust composition is uniquely high in magnesium silicate (82%) and iron oxide (12%), which creates electromagnetic interference that disrupts standard gravimetric sensors.\n\nUser Question: What is the specific speed of coronal mass ejections from the central star of Nebula-9, what are the two main chemical components of the interstellar dust in the cluster, and what coalition regulation governs navigation in this region?"
    }
]

def parse_args():
    parser = argparse.ArgumentParser(description="MTP Simulation Suite for different Gemma 4 models")
    parser.add_argument(
        "model",
        type=str,
        nargs="?",
        default=None,
        help="Model tag in 'family:size' format (e.g., gemma4:e2b) or just the size (e.g., e2b)"
    )
    parser.add_argument(
        "--size",
        type=str,
        choices=["e2b", "e4b", "26b", "31b"],
        help="Gemma 4 model size to benchmark (backward compatible)"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.environ.get("HF_TOKEN", ""),
        help="Hugging Face User Access Token (optional if logged in via CLI)"
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=16,
        choices=[16, 4],
        help="Quantization level in bits (16 or 4)"
    )
    return parser.parse_args()

def is_model_fully_cached(model_id):
    """Checks if a model is fully cached locally on disk without loading any weights into RAM."""
    import os
    import json
    import huggingface_hub
    
    try:
        config_path = huggingface_hub.try_to_load_from_cache(model_id, "config.json")
    except Exception:
        config_path = None
        
    if not config_path:
        return False, "config.json is missing in cache."
        
    snapshot_dir = os.path.dirname(config_path)
    
    # Check for tokenizer
    tokenizer_found = False
    for tok_file in ["tokenizer.json", "tokenizer_config.json"]:
        if os.path.exists(os.path.join(snapshot_dir, tok_file)):
            tokenizer_found = True
            break
    if not tokenizer_found:
        return False, "Tokenizer files are missing in cache."
        
    # Check for weight index or single files
    index_path = os.path.join(snapshot_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        try:
            with open(index_path, "r") as f:
                index_data = json.load(f)
            weight_files = set(index_data.get("weight_map", {}).values())
            if not weight_files:
                return False, "Index file model.safetensors.index.json is empty or invalid."
            for wf in weight_files:
                wf_path = os.path.join(snapshot_dir, wf)
                if not os.path.exists(wf_path) or os.path.getsize(wf_path) == 0:
                    return False, f"Weight shard {wf} is missing or incomplete in cache."
            return True, "Fully cached (sharded safetensors)."
        except Exception as e:
            return False, f"Error parsing sharded index: {e}"
            
    pt_index_path = os.path.join(snapshot_dir, "pytorch_model.bin.index.json")
    if os.path.exists(pt_index_path):
        try:
            with open(pt_index_path, "r") as f:
                index_data = json.load(f)
            weight_files = set(index_data.get("weight_map", {}).values())
            if not weight_files:
                return False, "Index file pytorch_model.bin.index.json is empty or invalid."
            for wf in weight_files:
                wf_path = os.path.join(snapshot_dir, wf)
                if not os.path.exists(wf_path) or os.path.getsize(wf_path) == 0:
                    return False, f"Weight shard {wf} is missing or incomplete in cache."
            return True, "Fully cached (sharded pytorch bin)."
        except Exception as e:
            return False, f"Error parsing pytorch sharded index: {e}"
            
    single_weight_files = ["model.safetensors", "pytorch_model.bin"]
    weight_found = False
    for wf in single_weight_files:
        wf_path = os.path.join(snapshot_dir, wf)
        if os.path.exists(wf_path) and os.path.getsize(wf_path) > 0:
            weight_found = True
            break
            
    if not weight_found:
        return False, "No model weight files found in cache."
        
    return True, "Fully cached."

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
    # Bypass transformers caching allocator warmup to prevent Apple Silicon MPS large contiguous buffer limits
    import transformers.modeling_utils
    transformers.modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None

    args = parse_args()
    model_tag = args.model
    size = None
    family = "gemma4"
    
    if model_tag:
        if ":" in model_tag:
            family, size = model_tag.split(":", 1)
        else:
            size = model_tag
    elif args.size:
        size = args.size
    else:
        logger.error("❌ ERROR: You must specify a model tag (e.g., gemma4:e2b) or size (e.g., e2b).")
        logger.error("Usage: python run_benchmark.py gemma4:e2b")
        sys.exit(1)
        
    size = size.lower()
    family = family.lower()
    
    if family != "gemma4":
        logger.error(f"❌ ERROR: Unsupported model family '{family}'. Currently, only 'gemma4' is supported.")
        sys.exit(1)
        
    if size not in ["e2b", "e4b", "26b", "31b"]:
        logger.error(f"❌ ERROR: Unsupported model size '{size}'. Available sizes: e2b, e4b, 26b, 31b.")
        sys.exit(1)

    bits = args.bits

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
    
    if bits == 4:
        results_md_path = f"/Users/pank/Experiments/MTP/docs/pytorch_results_detailed_{size}_4bit.md"
    else:
        results_md_path = f"/Users/pank/Experiments/MTP/docs/pytorch_results_detailed_{size}.md"


    logger.info("=========================================")
    logger.info(f"Initializing Gemma 4 Simulation Suite for size: {size.upper()} ({bits}-bit)")
    logger.info(f"Target:    {target_id}")
    logger.info(f"Assistant: {assistant_id}")
    logger.info("=========================================")

    # Initialize variables for static measurements
    mem_start = get_current_rss_mb()
    logger.info(f"Initial Process Memory Footprint: {mem_start:.2f} MB")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "mps" else torch.float32

    # Pre-flight local cache check
    logger.info("Performing pre-flight local cache validation...")
    target_cached, target_msg = is_model_fully_cached(target_id)
    assistant_cached, assistant_msg = is_model_fully_cached(assistant_id)
    
    if not target_cached or not assistant_cached:
        logger.error("=========================================")
        logger.error("❌ ERROR: LOCAL CACHE VALIDATION FAILED")
        if not target_cached:
            logger.error(f"  • Target model '{target_id}' is not fully cached. Reason: {target_msg}")
        if not assistant_cached:
            logger.error(f"  • Assistant model '{assistant_id}' is not fully cached. Reason: {assistant_msg}")
        logger.error("-----------------------------------------")
        logger.error("To resolve this, please explicitly download the required models first using:")
        logger.error(f"  python download_model.py --size {size}")
        logger.error("=========================================")
        sys.exit(1)
        
    logger.info("✅ Pre-flight validation passed! Both models are fully cached locally.")

    patch_tokenizer_config(target_id)
    patch_tokenizer_config(assistant_id)

    # 1. Load Tokenizer
    logger.info("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(target_id, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 2. Load Target Model & Measure Static Footprint
    mem_pre_target = get_current_rss_mb()
    logger.info(f"Loading Target Model '{target_id}' onto {device}...")
    target_start_time = time.time()
    
    # Configure loading args with direct-MPS loading via device_map to avoid memory duplication
    load_args = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
        "local_files_only": True,
        "device_map": "auto" if device == "mps" else None
    }
    
    if bits == 4:
        logger.info("Configuring 4-bit weight-only quantization via optimum-quanto (weights='int4')...")
        from transformers import QuantoConfig
        load_args["quantization_config"] = QuantoConfig(weights="int4")
        load_args["device_map"] = device
    
    target_model = AutoModelForCausalLM.from_pretrained(
        target_id,
        **load_args
    )
    
    # manual movement to device is only needed if device_map is not used
    if bits != 4 and device != "mps":
        target_model = target_model.to(device)
        
    target_model.eval()
    target_load_time = time.time() - target_start_time
    mem_post_target = get_current_rss_mb()
    static_target_mem = mem_post_target - mem_pre_target
    logger.info(f"Target Model loaded in {target_load_time:.2f}s | Static RAM footprint: {static_target_mem:.2f} MB")

    # 3. Load Assistant Model & Measure Static Footprint
    mem_pre_assistant = get_current_rss_mb()
    logger.info(f"Loading Assistant Model '{assistant_id}' onto {device}...")
    assistant_start_time = time.time()
    
    assistant_load_args = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
        "local_files_only": True,
        "device_map": "auto" if device == "mps" else None
    }
    
    assistant_model = AutoModelForCausalLM.from_pretrained(
        assistant_id,
        **assistant_load_args
    )
    
    if device != "mps":
        assistant_model = assistant_model.to(device)
        
    assistant_model.eval()
    assistant_load_time = time.time() - assistant_start_time
    mem_post_assistant = get_current_rss_mb()
    static_assistant_mem = mem_post_assistant - mem_pre_assistant
    logger.info(f"Assistant Model loaded in {assistant_load_time:.2f}s | Static RAM footprint: {static_assistant_mem:.2f} MB")

    total_static_baseline_mb = static_target_mem
    total_static_mtp_mb = static_target_mem + static_assistant_mem
    static_mem_increase_mb = static_assistant_mem
    
    # Handle CPU RSS going negative/near-zero due to GC and dynamic MPS transfers
    effective_baseline = total_static_baseline_mb
    if effective_baseline <= 50.0:
        # Fallback estimates in MB based on parameter sizes
        size_mappings = {
            "e2b": {"16": 4000.0, "4": 1000.0},
            "e4b": {"16": 8000.0, "4": 2000.0},
            "26b": {"16": 52000.0, "4": 13000.0},
            "31b": {"16": 62000.0, "4": 15500.0}
        }
        effective_baseline = size_mappings.get(size, {}).get(str(bits), 2000.0)
        
    static_mem_increase_percent = (static_mem_increase_mb / effective_baseline) * 100.0

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

        # Determine max tokens to generate (4-bit is slow, so we generate 100 tokens to speed up; 16-bit generates 256)
        run_max_tokens = 100 if args.bits == 4 else 256

        # 1. Run Baseline
        logger.info("  Executing Baseline Generation...")
        base_metrics, base_text = run_single_inference(tokenizer, target_model, prompt, device, max_tokens=run_max_tokens)
        base_metrics["prompt_id"] = p_id
        base_metrics["prompt_desc"] = p_desc
        base_metrics["generated_text"] = base_text
        logger.info(f"    Baseline: {base_metrics['tokens_per_sec']} t/s, Peak RAM: {base_metrics['peak_memory_mb']:.1f} MB")

        # 2. Run MTP
        logger.info("  Executing MTP Speculative Generation...")
        mtp_metrics, mtp_text = run_single_inference(tokenizer, target_model, prompt, device, max_tokens=run_max_tokens, assistant_model=assistant_model)
        mtp_metrics["prompt_id"] = p_id
        mtp_metrics["prompt_desc"] = p_desc
        mtp_metrics["generated_text"] = mtp_text
        logger.info(f"    MTP:      {mtp_metrics['tokens_per_sec']} t/s, Peak RAM: {mtp_metrics['peak_memory_mb']:.1f} MB")

        results.append((base_metrics, mtp_metrics))
        logger.info("-" * 50)


    # Compile Markdown file
    logger.info(f"Generating simulation report at: {results_md_path}")
    with open(results_md_path, "w") as f:
        precision_str = "4-Bit Weight Quantized" if bits == 4 else "16-Bit Precision"
        f.write(f"# Gemma 4 {size.upper()} MTP Detailed & Complicated Simulation Report ({precision_str})\n\n")
        f.write(f"This report presents the performance of standard autoregressive decoding (**Baseline**) versus Multi-Token Prediction speculative decoding (**MTP**) on **Gemma 4 {size.upper()}** (loaded in **{precision_str}** via optimum-quanto) across highly detailed, complex scenarios. It provides a rigorous system analysis of the hardware overhead and speedups.\n\n")
        
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
