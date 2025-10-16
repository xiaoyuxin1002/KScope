#!/bin/bash

stdbuf -oL -eL python extract.py --model=Llama-3.1-8B-Instruct >&2
stdbuf -oL -eL python cluster.py --model=Llama-3.1-8B-Instruct >&2
stdbuf -oL -eL python status_open.py --model=Llama-3.1-8B-Instruct >&2