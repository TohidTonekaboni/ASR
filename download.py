import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv()

# The model is public and the proxy rejects the HF token with 401, so send no token.
token = False

# Set your custom Hugging Face proxy endpoint
endpoint = os.environ.get("HF_ENDPOINT", "https://new-pkg.mydigipay.info/repository/huggingface-proxy").rstrip("/")

# Apply the custom endpoint globally
os.environ["HF_ENDPOINT"] = endpoint

# Download model through the proxy
snapshot_download(
    repo_id="Reza2kn/Shenava-Koochik-v1.0",
    local_dir="./model",
    token=token,
    endpoint=endpoint
)

print("Downloaded from proxy:", endpoint)