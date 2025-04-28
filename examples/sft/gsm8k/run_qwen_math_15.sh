set -e

if [ "$#" -lt 2 ]; then
    echo "Usage: run_qwen_05_sp2.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

export WANDB_MODE="online"

# model=Qwen/Qwen2.5-0.5B-Instruct
# model=HuggingFaceTB/SmolLM2-135M-Instruct
# model=Qwen/Qwen2.5-Math-1.5B-Instruct
model=meta-llama/Llama-3.2-1B-Instruct


experiment_name=gsm8k-sft-$model-liger

# Create the list of test parquet files
test_files="[$HOME/Jack/datasets/gsm8k/test.parquet]"

export VERL_SFT_LOGGING_LEVEL=INFO
torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/Jack/datasets/gsm8k/train.parquet \
    data.val_files=$test_files \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.train_batch_size=128 \
    optim.lr=5e-5 \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    model.partial_pretrain=/home/ubuntu/Jack/checkpoints/gsm8k-sft-meta-llama/Llama-3.2-1B-Instruct-liger/global_step_58\
    model.use_liger=true \
    trainer.default_local_dir=$save_path/$experiment_name \
    trainer.project_name=igsm-sft \
    trainer.experiment_name=$experiment_name \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=null $@ \
    validation.reward_fn.path="/home/ubuntu/CS839-Project/verl/verl/utils/reward_score/gsm8k.py" \
    validation.val_before_train=true \
    use_remove_padding=false 
