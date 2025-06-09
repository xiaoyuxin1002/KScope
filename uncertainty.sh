#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

stdbuf -oL -eL python entropy.py --model=meta-llama/Llama-3.1-8B-Instruct >&2