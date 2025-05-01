set -e

if [ "$#" -lt 2 ]; then
    echo "Usage: run_qwen_math_15_liger.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

export WANDB_MODE="disabled"

model=Qwen/Qwen2.5-0.5B-Instruct
# model=HuggingFaceTB/SmolLM2-135M-Instruct
# model=Qwen/Qwen2.5-7B-Instruct
# model=Qwen/Qwen2.5-Math-1.5B-Instruct

for round in {1..10}; do
    experiment_name=igsm-sft-$model-liger-round_$round-no_label
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
    for ((n=round; n<=round+24; n+=3)); do
        # Add comma separator for multiple files
        if [ -n "$test_files" ]; then
            test_files="$test_files,"
        fi
        test_files="$test_files$HOME/Jack/datasets/igsm/test_op_$n.parquet"
    done
    test_files="[$test_files]"

    # Create the list of train parquet files
    train_files=""
    for ((n=round; n<round+5; n+=1)); do
        # Add comma separator for multiple files
        if [ -n "$train_files" ]; then
            train_files="$train_files,"
        fi
        # if very first round, we use the original train file
        if [ "$n" -le 5 ]; then
            next_train_file=$HOME/Jack/datasets/igsm/train_op_$n.parquet
        else
            next_train_file=$HOME/Jack/datasets/igsm/gen/igsm-sft-$model-liger/train_op_$n.parquet
        fi
        train_files="$train_files$next_train_file"
    done
    train_files="[$train_files]"

    export WANDB_RUN_GROUP=$experiment_name

    torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
        -m verl.trainer.fsdp_sft_trainer \
        data.train_files=$train_files \
        data.val_files=$test_files \
        data.prompt_key=prompt \
        data.prompt_dict_keys=[] \
        data.response_key=response \
        data.response_dict_keys=[] \
        data.train_batch_size=64 \
        data.micro_batch_size_per_gpu=8 \
        optim.lr=5e-5 \
        model.partial_pretrain=$last_round_name \
        model.use_liger=true \
        trainer.resume_from_checkpoint=true \
        trainer.default_local_dir=$save_path/$experiment_name \
        trainer.project_name=igsm-sft \
        trainer.experiment_name=$experiment_name \
        trainer.logger=['console','wandb'] \
        trainer.default_hdfs_dir=null $@ \
        trainer.total_epochs=1 \
        validation.reward_fn.path="/home/ubuntu/CS839-Project/verl/verl/utils/reward_score/igsm.py" \
        validation.val_before_train=false \
        use_remove_padding=false

    python -m verl.trainer.main_generation \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=1 \
        data.path=$HOME/Jack/datasets/igsm/train_op_$(($round+5)).parquet \
        data.prompt_key=prompt \
        data.n_samples=1 \
        data.output_path=$HOME/Jack/datasets/igsm/gen/igsm-sft-$model-liger/train_op_$((round+5)).parquet \
        model.path=$save_path/$experiment_name \
        +model.trust_remote_code=True \
        rollout.temperature=1.0 \
        rollout.top_k=50 \
        rollout.top_p=0.7 \
        rollout.prompt_length=2048 \
        rollout.response_length=1024 \
        rollout.tensor_model_parallel_size=1 \
        rollout.gpu_memory_utilization=0.95
done
        # trainer.resume_path=$save_path/$experiment_name \
