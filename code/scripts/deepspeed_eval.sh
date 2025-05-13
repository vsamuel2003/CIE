#!/bin/bash

python requeue_failed_gpu.py
deepspeed  --master_port 29690 ../eval.py \
            --model_identifier llama3 \
            --model /data/models/huggingface/meta-llama/Meta-Llama-3-8B-Instruct \
            --bs 16 \
            --prediction_file_name full_finetune/llama3_epoch3 \
            --model_saved_dir /data/group_data/word_count/full/5e-6/checkpoint-5832 \
            --benchmark validation
            
