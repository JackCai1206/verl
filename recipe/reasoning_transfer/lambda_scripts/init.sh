if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # export PATH="$HOME/.local/bin:$PATH"
else
    echo "uv is already installed."
fi

uv venv --python=python3.10

source .venv/bin/activate
python -m ensurepip
uv pip install torch torchvision torchaudio
# uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --python-version 3.10 --only-binary=:all:

uv pip install wheel
uv pip install flash-attn==2.7.3 --no-build-isolation 
# python -m pip install flash-attn --index-url https://pypi.jetson-ai-lab.dev/sbsa/cu128
uv pip install flashinfer-python==0.2.2

uv pip install liger-kernel

uv pip install vllm==0.8.5
# python -m pip install xgrammar vllm --index-url https://pypi.jetson-ai-lab.dev/sbsa/cu128/

uv pip install -e .

uv pip install jupyter ipykernel matplotlib

uv pip install hf_transfer math_verify

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cudnn9-cuda-12
uv pip install --no-build-isolation transformer_engine[pytorch]
uv pip install megatron-core

# Allow login from freyr
echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIKKc20MGaVC5TKeF/bkpmPIbyD31oBqDiO94fcMSkiE2 jackcai1206@gmail.com" >> ~/.ssh/authorized_keys

wandb login $WANDB_API_KEY

git config --global user.name "jackcai1206"
git config --global user.email "jackcai1206@gmail.com"
