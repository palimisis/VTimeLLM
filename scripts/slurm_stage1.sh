#!/bin/bash
#SBATCH --partition=luke
#SBATCH --ntasks 1
#SBATCH --gres=gpu:0
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=100GB


set -e
# Activate Anaconda work environment
# source /home/$USER/minicoda3/etc/profile.d/conda.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda init bash
conda activate v-llm


MODEL_VERSION=vicuna-v1-5-7b
gpu_vis=0 # per_device_train_batch_size * gradient_accumulation_steps * n_gpus = 128
MASTER_PORT=29570


deepspeed --include localhost:$gpu_vis --master_port $MASTER_PORT vtimellm/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ./checkpoints/vicuna-7b-v1.5 \
    --version plain \
    --data_path ./data/blip_laion_cc_sbu_558k.json \
    --feat_folder /path/to/stage1_feat \
    --tune_mm_mlp_adapter True \
    --output_dir ./checkpoints/vtimellm-$MODEL_VERSION-stage1 \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb 