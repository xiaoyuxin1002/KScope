#!/bin/bash

stdbuf -oL -eL python status.py --dataset=Hemonc --model=meta-llama/Llama-3.1-8B-Instruct >&2