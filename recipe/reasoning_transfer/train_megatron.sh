set -x

# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS
# export RAY_DEBUG_POST_MORTEM=1 
export CUDA_DEVICE_MAX_CONNECTIONS=1 # For megatron communication/computation overlapping

model_name="Qwen/Qwen3-1.7B"
experiment_name=$model_name

train_files="['$HOME/Jack/data/verl/jackcai1206_sudoku_easy2hard/train.parquet']"
val_files="['$HOME/Jack/data/verl/jackcai1206_sudoku_easy2hard/test.parquet', '$HOME/Jack/data/verl/HuggingFaceH4_aime_2024/train.parquet', '$HOME/Jack/data/verl/HuggingFaceH4_MATH-500/test.parquet', '$HOME/Jack/data/verl/MathArena_aime_2025/train.parquet']"

tp_size=1

export VERL_LOGGING_LEVEL=INFO
export VLLM_LOGGING_LEVEL=INFO

python3 -m verl.trainer.main_ppo --config-path=config \
    --config-name='ppo_megatron_trainer.yaml'\
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$val_files" \
    data.train_batch_size=512 \
    data.max_prompt_length=650 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=16 \
    data.truncation='error' \
    actor_rollout_ref.model.path="$model_name" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${tp_size} \
    actor_rollout_ref.actor.use_dynamic_bsz=False\
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tp_size} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${tp_size} \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='reasoning_transfer' \
    trainer.experiment_name="$experiment_name" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 \
    custom_reward_function.path=recipe/reasoning_transfer/reward_function.py \
    custom_reward_function.name=compute_score $@