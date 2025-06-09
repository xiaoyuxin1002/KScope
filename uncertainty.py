import os
import time
import json
import argparse
import warnings
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def myprint(text):
    print(time.strftime("%Y %b %d %a, %H:%M:%S: ", time.localtime()) + text, flush=True)

def get_info(text):
    input = tokenizer(text, truncation=True, max_length=min(model.config.max_position_embeddings, 30000), return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model(**input, use_cache=True, output_hidden_states=False, output_attentions=False)
    probs = torch.clamp(F.softmax(output.logits[0], dim=-1), min=1e-12)
    e = -torch.sum(probs * torch.log(probs), dim=1).mean().item()
    loglikelihood = F.log_softmax(output.logits[0], dim=-1)
    p = torch.exp(-loglikelihood[:-1][torch.arange(input.input_ids.shape[1]-1), input.input_ids[0,1:]].mean()).item()
    return e, p

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
args = parser.parse_args()
warnings.filterwarnings("ignore")
myprint(f'Computing the Entropy and Perplexity of {args.model}')

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map='auto')

cwd = os.getcwd()
keys = ['question 1', 'evidence']
datas = ['Hemonc', 'PubMedQA', 'HotpotQA', 'NQ']
        
entropy, perplexity = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
for data in datas:
    dataset = pd.read_csv(f'{cwd}/Data/Input/{data}.csv')
    for key in keys:
        for text in tqdm(dataset[key], desc=f'{data} - {key}'):
            e, p = get_info(text)
            entropy[data][key].append(e); perplexity[data][key].append(p)
json.dump(entropy, open(f'{cwd}/Data/Uncertainty/{args.model.split("/")[-1]}_entropy.json', 'w'), indent=4)
json.dump(perplexity, open(f'{cwd}/Data/Uncertainty/{args.model.split("/")[-1]}_perplexity.json', 'w'), indent=4)