#!/bin/bash

export NCCL_P2P_DISABLE=1
deepspeed  --master_port 29690 ../verbosity-finetune-deepspeed.py \
            --model_type gemma \
            --model_name /data/group_data/models--google--gemma-7b-it/snapshots/9c5798d27f588501ce1e108079d2a19e4c3a2353 \
            --wandb_run_name gemma-deepspeed-5e-5_7\
            --do_train \
            --num_train_epochs 7 \
            --save_strategy epoch \
            --learning_rate 5e-5 \
            --per_device_train_batch_size 1 \
            --gradient_accumulation_steps 4 \
            --optim paged_adamw_32bit \
            --max_train_samples 100000 \
            --weight_decay 0.001 \
            --ddp_find_unused_parameters false \
            --save_total_limit 3 \
            --output_dir /data/group_data/word_count/full/gemma/5e-5_7 \
            --max_grad_norm 0.3 \
            --warmup_ratio 0.03 \
            --fp16 False \
            --bf16 True \
            --lr_scheduler_type cosine \
            --deepspeed ./deepspeed_config.json \
            --save_safetensors False
