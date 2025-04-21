set -ex

if [ "$#" -lt 2 ]; then
    echo "Usage: run_qwen_05_sp2.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

export WANDB_MODE="disabled"

model=Qwen/Qwen2.5-0.5B-Instruct
# model=HuggingFaceTB/SmolLM2-135M-Instruct

for round in {1..5}; do
    experiment_name=igsm-sft-$model-liger-round_$round

    torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
        -m verl.trainer.fsdp_sft_trainer \
        data.train_files=$HOME/Jack/datasets/igsm/train_$round.parquet \
        data.val_files=$HOME/Jack/datasets/igsm/test_$round.parquet \
        data.prompt_key=extra_info \
        data.response_key=extra_info \
        optim.lr=1e-4 \
        data.prompt_dict_keys=['full_question'] \
        +data.response_dict_keys=['full_solution'] \
        data.micro_batch_size=4 \
        model.partial_pretrain=$model \
        model.use_liger=False \
        trainer.default_local_dir=$experiment_name/$save_path \
        trainer.project_name=igsm-sft \
        trainer.experiment_name=$experiment_name \
        trainer.logger=['console','wandb'] \
        trainer.default_hdfs_dir=null $@ \
        trainer.resume_path=$experiment_name/$save_path \
        trainer.val_freq_steps=50 \
        custom_reward_function.path="/home/ubuntu/Jack/CS839-Project/verl/verl/utils/reward_score/igsm.py" \
        use_remove_padding=false
done
