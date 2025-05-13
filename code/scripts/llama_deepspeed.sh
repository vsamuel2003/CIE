#!/bin/bash

export NCCL_P2P_DISABLE=1
deepspeed  --master_port 29690 ../verbosity-finetune-deepspeed.py \
            --model_type llama3 \
            --model_name /data/models/huggingface/meta-llama/Meta-Llama-3-8B-Instruct \
            --wandb_run_name word-count-full-deepspeed-5e-6_7\
            --do_train \
            --num_train_epochs 1 \
            --save_strategy epoch \
            --learning_rate 5e-6 \
            --per_device_train_batch_size 1 \
            --gradient_accumulation_steps 4 \
            --optim paged_adamw_32bit \
            --max_train_samples 100000 \
            --weight_decay 0.001 \
            --ddp_find_unused_parameters false \
            --save_total_limit 3 \
            --output_dir /data/group_data/word_count/full/5e-6_test \
            --max_grad_norm 0.3 \
            --warmup_ratio 0.03 \
            --fp16 False \
            --bf16 True \
            --lr_scheduler_type cosine \
            --deepspeed ./deepspeed_config.json \
            
