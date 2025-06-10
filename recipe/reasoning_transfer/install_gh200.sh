sudo apt-get update && sudo apt-get install -y libnuma-dev
# uv pip install triton==3.3.0 --find-links https://download.pytorch.org/whl/triton/
# uv pip install https://pypi.jetson-ai-lab.dev/sbsa/cu128/+f/460/dd36ac5d2a919/torch-2.7.0-cp312-cp312-linux_aarch64.whl#sha256=460dd36ac5d2a91952ecc436108b0a78524a91b33701e27394c581cb5f8a7abc
# uv pip install https://pypi.jetson-ai-lab.dev/sbsa/cu128/+f/e68/68895880e5179/vllm-0.8.5+cu128-cp312-cp312-linux_aarch64.whl
# uv pip install https://pypi.jetson-ai-lab.dev/sbsa/cu128/+f/573/afdcbdbef544e/flash_attn-2.7.4.post1-cp312-cp312-linux_aarch64.whl
# uv pip install https://pypi.jetson-ai-lab.dev/sbsa/cu128/+f/d59/4d5bde219df1f/flashinfer_python-0.2.5-cp312-cp312-linux_aarch64.whl

sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install --only-upgrade libstdc++6

python -m pip install xgrammar torch torchvision torchaudio vllm==0.8.5 --index-url https://pypi.jetson-ai-lab.dev/sbsa/cu128/
python -m pip install flash-attn flashinfer-python --index-url https://pypi.jetson-ai-lab.dev/sbsa/cu128/
uv pip install --no-cache-dir tensordict torchdata
uv pip install "transformers[hf_xet]>=4.51.0" accelerate datasets peft hf-transfer     "numpy<2.0.0" "pyarrow>=15.0.0" pandas     ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler     pytest py-spy pyext pre-commit ruff
uv pip install "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"

uv pip install --no-deps -e .

uv pip install math_verify
