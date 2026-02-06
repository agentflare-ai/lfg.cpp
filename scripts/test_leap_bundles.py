import os
import json
import requests
import subprocess
import sys
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "LiquidAI/LeapBundles"
TEST_BINARY = "./build/bin/test_model_capabilities"
MODELS_DIR = "test_models"

api = HfApi()

def download_file(url, dest):
    print(f"Downloading {url} to {dest}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--vision-only", action="store_true")
    parser.add_argument("--audio-only", action="store_true")
    parser.add_argument("--exclude-qwen", action="store_true", default=True)
    args = parser.parse_args()

    if not os.path.exists(TEST_BINARY):
        print(f"Error: {TEST_BINARY} not found. Build the project first.")
        sys.exit(1)

    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Recursively list all files in the repo
    print(f"Listing files in {REPO_ID}...")
    files = list(api.list_repo_tree(REPO_ID, recursive=True))
    
    results = {}
    processed_count = 0

    for file_info in files:
        if args.limit is not None and processed_count >= args.limit:
            break
        
        path = file_info.path
        
        if path.endswith(".json") and "/" in path:
            if os.path.basename(path) == "schema.json":
                continue
            
            if args.exclude_qwen and "Qwen" in path:
                print(f"Skipping Qwen model: {path}")
                continue

            if "Audio" in path:
                print(f"Skipping Audio model (arch mismatch): {path}")
                continue

            if args.vision_only and "VL" not in path:
                continue

            if args.audio_only and "Audio" not in path:
                continue
                
            print(f"Processing config: {path}")
            try:
                config_path = hf_hub_download(repo_id=REPO_ID, filename=path, local_dir=MODELS_DIR)
                
                with open(config_path, "r") as f:
                    config = json.load(f)
                
                params = config.get("load_time_parameters", {})
                gguf_url = params.get("model")
                mmproj_url = params.get("multimodal_projector")

                if gguf_url and gguf_url.endswith(".gguf"):
                    model_name = os.path.basename(gguf_url)
                    model_dest = os.path.join(MODELS_DIR, model_name)
                    
                    if not os.path.exists(model_dest):
                        download_file(gguf_url, model_dest)
                    
                    cmd = [TEST_BINARY, "--model", model_dest]
                    
                    mmproj_dest = None
                    if mmproj_url:
                        mmproj_name = os.path.basename(mmproj_url)
                        mmproj_dest = os.path.join(MODELS_DIR, mmproj_name)
                        if not os.path.exists(mmproj_dest):
                            download_file(mmproj_url, mmproj_dest)
                        cmd.extend(["--mmproj", mmproj_dest])
                        # If vision, we might need a dummy image
                        if "VL" in path and os.path.exists("image.png"):
                            cmd.extend(["--image", "image.png"])
                        # If audio, add dummy audio flag
                        if "Audio" in path:
                            cmd.extend(["--audio", "dummy.wav"])

                    print(f"Running: {' '.join(cmd)}")
                    try:
                        subprocess.check_call(cmd)
                        results[path] = True
                    except subprocess.CalledProcessError:
                        results[path] = False
                    
                    processed_count += 1
                    # Clean up
                    if os.path.exists(model_dest): os.remove(model_dest)
                    if mmproj_dest and os.path.exists(mmproj_dest): os.remove(mmproj_dest)
                else:
                    print(f"Skipping {path}: No GGUF URL found.")
            except Exception as e:
                print(f"Error processing {path}: {e}")
                results[path] = False
    
    print("\nTest Summary:")
    for path, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"{path}: {status}")

if __name__ == "__main__":
    main()
