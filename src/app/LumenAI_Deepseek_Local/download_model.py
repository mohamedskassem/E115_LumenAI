import os
from huggingface_hub import snapshot_download

MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct"
MODEL_DIR = "./models/deepseek-coder-1.3b-instruct"

# Create the models directory if it doesn't exist
if not os.path.exists("./models"):
    os.makedirs("./models")
    print("Created 'models' directory.")

if not os.path.exists(MODEL_DIR):
    print(f"Downloading model {MODEL_NAME} to {MODEL_DIR}...")
    snapshot_download(
        repo_id=MODEL_NAME,
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.safetensors"] #remove this line if you want the safetensors file.
    )
    print("Model download complete.")
else:
    print(f"Model already exists at {MODEL_DIR}. Skipping download.")