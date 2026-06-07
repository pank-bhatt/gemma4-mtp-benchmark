import os
import sys
import argparse
import logging


import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(config.LOG_DIR, "models.log"), mode="a")
    ]
)
logger = logging.getLogger("ModelDownloader")

def parse_args():
    parser = argparse.ArgumentParser(description="Gemma 4 explicit model downloader and cacher")
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
        help="Gemma 4 model size to download (backward compatible)"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.environ.get("HF_TOKEN", ""),
        help="Hugging Face User Access Token (optional if logged in via CLI)"
    )
    parser.add_argument(
        "--mlx",
        action="store_true",
        help="Download Apple Silicon MLX community models instead of PyTorch models"
    )
    return parser.parse_args()

def check_hf_access(model_id, token):
    """Verifies Hugging Face hub access before starting download."""
    from huggingface_hub import HfApi
    try:
        api = HfApi(token=token if token else None)
        api.model_info(model_id)
        logger.info(f"✅ HF Access Verified for: {model_id}")
    except Exception as e:
        logger.error(f"❌ HF Model Access Error: Gated model access is required for {model_id}.")
        logger.error("Please run `huggingface-cli login` or pass a valid `--hf-token`.")
        sys.exit(1)

def main():
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
        logger.error("Usage: python download_model.py gemma4:e2b")
        sys.exit(1)
        
    size = size.lower()
    family = family.lower()
    
    if family != "gemma4":
        logger.error(f"❌ ERROR: Unsupported model family '{family}'. Currently, only 'gemma4' is supported.")
        sys.exit(1)
        
    if size not in ["e2b", "e4b", "26b", "31b"]:
        logger.error(f"❌ ERROR: Unsupported model size '{size}'. Available sizes: e2b, e4b, 26b, 31b.")
        sys.exit(1)

    token = args.hf_token if args.hf_token else None

    # Map size to Hugging Face model IDs
    if args.mlx:
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
        targets = [model_mappings[size]["target_4bit"], model_mappings[size]["target_16bit"]]
        assistant_id = model_mappings[size]["assistant"]

        logger.info("=========================================")
        logger.info(f"Starting explicit MLX download for {family}:{size}")
        logger.info(f"Targets:    {targets}")
        logger.info(f"Assistant:  {assistant_id}")
        logger.info("=========================================")

        # Verify access
        logger.info("Verifying Hugging Face access...")
        for target_id in targets:
            check_hf_access(target_id, token)
        check_hf_access(assistant_id, token)

        from huggingface_hub import snapshot_download

        # Download targets
        for target_id in targets:
            logger.info(f"Downloading MLX Target Model '{target_id}' via snapshot_download (zero-RAM footprint)...")
            target_dir = snapshot_download(
                repo_id=target_id,
                token=token,
                repo_type="model",
                max_workers=8
            )
            logger.info(f"✅ Target model {target_id} successfully cached at: {target_dir}")

        # Download assistant
        logger.info(f"Downloading MLX Assistant Model '{assistant_id}' via snapshot_download (zero-RAM footprint)...")
        assistant_dir = snapshot_download(
            repo_id=assistant_id,
            token=token,
            repo_type="model",
            max_workers=8
        )
        logger.info(f"✅ Assistant model {assistant_id} successfully cached at: {assistant_dir}")

        logger.info("=========================================")
        logger.info(f"SUCCESS: {family.upper()}:{size.upper()} (MLX models) is now fully downloaded and cached locally!")
        logger.info("=========================================")

    else:
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

        logger.info("=========================================")
        logger.info(f"Starting explicit download for {family}:{size}")
        logger.info(f"Target:    {target_id}")
        logger.info(f"Assistant: {assistant_id}")
        logger.info("=========================================")

        # Verify access to both models
        logger.info("Verifying Hugging Face gated access...")
        check_hf_access(target_id, token)
        check_hf_access(assistant_id, token)

        # 1. Download target model and tokenizer
        logger.info(f"Downloading Target Model '{target_id}' via snapshot_download (zero-RAM footprint)...")
        from huggingface_hub import snapshot_download
        
        target_dir = snapshot_download(
            repo_id=target_id,
            token=token,
            repo_type="model",
            max_workers=8
        )
        logger.info(f"✅ Target model {target_id} successfully cached at: {target_dir}")

        # 2. Download assistant model
        logger.info(f"Downloading Assistant Model '{assistant_id}' via snapshot_download (zero-RAM footprint)...")
        assistant_dir = snapshot_download(
            repo_id=assistant_id,
            token=token,
            repo_type="model",
            max_workers=8
        )
        logger.info(f"✅ Assistant model {assistant_id} successfully cached at: {assistant_dir}")

        logger.info("=========================================")
        logger.info(f"SUCCESS: {family.upper()}:{size.upper()} is now fully downloaded and cached locally!")
        logger.info(f"You can now run benchmarks offline using:")
        logger.info(f"  python run_benchmark.py {family}:{size}")
        logger.info("=========================================")

if __name__ == "__main__":
    main()
