#!/bin/bash

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=0,1,2,3

datasets=("Hemonc" "PubMedQA" "NQ" "HotpotQA")
models_s=("meta-llama/Llama-3.2-3B-Instruct" "google/gemma-2-2b-it" "Qwen/Qwen2.5-3B-Instruct")
models_m=("meta-llama/Llama-3.1-8B-Instruct" "google/gemma-2-9b-it" "Qwen/Qwen2.5-7B-Instruct")
models_l=("meta-llama/Llama-3.3-70B-Instruct" "google/gemma-2-27b-it" "Qwen/Qwen2.5-14B-Instruct")

for dataset in ${datasets[@]}
do
    for model in ${models_s[@]}
    do
        stdbuf -oL -eL python response.py --dataset=${dataset} --model=${model} --num_gpus=1 >&2
    done
    for model in ${models_m[@]}
    do
        stdbuf -oL -eL python response.py --dataset=${dataset} --model=${model} --num_gpus=1 >&2
    done
    for model in ${models_l[@]}
    do
        stdbuf -oL -eL python response.py --dataset=${dataset} --model=${model} --num_gpus=4 >&2
    done
done
