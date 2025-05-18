#!/bin/bash
    
# --num_gpus=4
# --bits 4 \

model_max_length=16384   
model_name_or_path='microsoft/Phi-3-mini-128k-instruct'
data_path="dataset/"
url="http://128.213.11.13:9999/blazegraph/namespace/kb"

deepspeed --num_gpus=2 train.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $model_name_or_path \
    --data_path $data_path \
    --url $url \
    --bf16 True \
    --lora_enable True \
    --output_dir ./checkpoints/KGQA-Rec/ \
    --num_train_epochs 4 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --train_type 2 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length $model_max_length \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none