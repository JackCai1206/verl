export CUDA_VISIBLE_DEVICES=0
set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_qwen_05_sp2.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/Jack/datasets/igsm/train.parquet \
    data.val_files=$HOME/Jack/datasets/igsm/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    optim.lr=1e-4 \
    data.prompt_dict_keys=['full_question'] \
    +data.response_dict_keys=['full_solution'] \
    data.micro_batch_size=4 \
    model.partial_pretrain=Qwen/Qwen2.5-0.5B-Instruct \
    model.use_liger=True \
    trainer.default_local_dir=$save_path \
    trainer.project_name=igsm-sft \
    trainer.experiment_name=igsm-sft-qwen-2.5-0.5b-instruct-liger \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=null $@ \
    use_remove_padding=true
