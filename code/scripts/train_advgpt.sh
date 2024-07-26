#!/bin/bash

deepspeed --include localhost:0,1 --master_addr 127.0.0.1 --master_port 28457 train_sft.py \
    --model openllama_peft \
    --stage 1\
    --data_path  ../data/advgpt_visual_instruction_imagenet_dataset/advgpt_visual_instruction_imagenet_data.json\
    --image_root_path ../data/advgpt_visual_instruction_imagenet_dataset/images/\
    --imagebind_ckpt_path ../pretrained_ckpt/imagebind_ckpt/\
    --vicuna_ckpt_path ../pretrained_ckpt/vicuna_ckpt/7b_v0/\
    --max_tgt_len 64\
    --save_path  ./ckpt/advgpt_imagenet_7b_peft/\
    --log_path ./ckpt/advgpt_imagenet_7b_peft/log_rest/
