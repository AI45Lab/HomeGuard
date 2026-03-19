export PYTHONPATH=$PYTHONPATH:$(pwd)/src
export WANDB_PROJECT="HomeGuard"

# Create log directory and set log file
mkdir -p log
MODEL_NAME="qwen3-vl-8b-thinking"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="log/${MODEL_NAME}_${TIMESTAMP}.txt"

set -x

export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
# export NCCL_DEBUG=INFO

{
    llamafactory-cli train \
    --model_name_or_path ${MODEL_PATH} \
    --trust_remote_code \
    --stage sft \
    --do_train \
    --finetuning_type lora \
    --lora_rank 8 \
    --lora_target all \
    --dataset hazard_detection_step_cot \
    --template qwen3_vl \
    --cutoff_len 4096 \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 4 \
    --output_dir saves/${MODEL_NAME}_lora_step_sft \
    --logging_steps 10 \
    --save_steps 500 \
    --plot_loss \
    --overwrite_output_dir \
    --save_only_model false \
    --report_to wandb \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --num_train_epochs 2.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 \
    --ddp_timeout 180000000
} 2>&1 | tee -a "${LOG_FILE}"