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
# model=Qwen/Qwen2.5-7B-Instruct
model=Qwen/Qwen2.5-Math-1.5B-Instruct

for round in {1..5}; do
    experiment_name=igsm-sft-$model-liger-round_$round
    if [ "$round" -eq 1 ]; then
        last_round_name=$model
    else
        prev_round=$((round-1))
        last_round_name=$save_path/igsm-sft-$model-liger-round_$prev_round
        # Find the latest checkpoint folder with the biggest global step
        last_round_name=$(find "$last_round_name" -maxdepth 1 -type d -name "global_step_*" | sort -t_ -k3 -n | tail -n 1)
        if [ -z "$last_round_name" ]; then
            echo "Error: Could not find checkpoint folder in $last_round_name"
            exit 1
        fi
    fi

    # Create the list of test parquet files
    test_files=""
    for ((n=round; n<=round+4; n++)); do
        # Add comma separator for multiple files
        if [ -n "$test_files" ]; then
            test_files="$test_files,"
        fi
        test_files="$test_files$HOME/Jack/datasets/igsm/test_op_$n.parquet"
    done
    test_files="[$test_files]"

    torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
        -m verl.trainer.fsdp_sft_trainer \
        data.train_files=$HOME/Jack/datasets/igsm/train_$round.parquet \
        data.val_files=$test_files \
        data.prompt_key=extra_info \
        data.response_key=extra_info \
        data.train_batch_size=128 \
        optim.lr=1e-4 \
        data.prompt_dict_keys=['full_question'] \
        +data.response_dict_keys=['full_solution'] \
        model.partial_pretrain=$save_path/$experiment_name/global_step_372 \
        model.use_liger=true \
        trainer.default_local_dir=$save_path/$experiment_name \
        trainer.project_name=igsm-sft \
        trainer.experiment_name=$experiment_name \
        trainer.logger=['console','wandb'] \
        trainer.default_hdfs_dir=null $@ \
        validation.reward_fn.path="/home/ubuntu/CS839-Project/verl/verl/utils/reward_score/igsm.py" \
        validation.reward_fn.name="compute_score" \
        validation.val_before_train=false \
        use_remove_padding=false 
done
        # trainer.resume_path=$save_path/$experiment_name \
