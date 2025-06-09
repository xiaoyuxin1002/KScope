import re
import os
import json
import time
import argparse
import warnings
import functools
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import binom, binomtest, chi2, multinomial 


def myprint(text):
    print(time.strftime("%Y %b %d %a, %H:%M:%S: ", time.localtime()) + text, flush=True)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Hemonc')
parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
args = parser.parse_args()
warnings.filterwarnings("ignore")
myprint(f'Characterize the knowledge status of {args.model} on {args.dataset}')

cwd = os.getcwd()
pattern = r"Option\s+\[?(\d+)\]?"
dataset = pd.read_csv(f'{cwd}/Data/Input/{args.dataset}.csv')
responses = json.load(open(f'{cwd}/Data/Output/{args.dataset}/{args.model.split("/")[-1]}.json'))

myprint('Step 0: Extract Prediction')
idx2predicts = {}
for idx, response in enumerate(responses):
    predict_mem, predict_con = [], []
    for each_mem, each_con in zip(response['memory'], response['context']):
        match_mem, match_con = re.search(pattern, each_mem), re.search(pattern, each_con)
        predict_mem.append(int(match_mem.group(1)) if match_mem else 0)
        predict_con.append(int(match_con.group(1)) if match_con else 0)
    idx2predicts[idx] = {'memory':np.array(predict_mem), 'context':np.array(predict_con)}
idx2answer = {idx:row['answer'] for idx, row in dataset.iterrows()}

myprint('Step 1: Test Refusal')
alpha, num_resp, num_opt = 0.05, 100, 3
thres_refusal = binom.isf(alpha, num_resp, 0.5)
idx_refusal, idx2counters = defaultdict(set), defaultdict(dict)
for idx, preds in idx2predicts.items():
    for key in ['memory', 'context']:
        if np.sum(preds[key]==0) >= thres_refusal: idx_refusal[key].add(idx)
        idx2counters[idx][key] = np.array([np.sum(preds[key]==option) for option in range(1, num_opt+1)])
        
myprint('Step 2: Test Random Guessing')
def multinom_test(obs):

    @functools.cache
    def find_vectors(_num_obs, _num_opt):
        mat = _num_obs
        if _num_opt != 1:
            mat = np.zeros((1, _num_opt-1))
            for i in range(1, _num_obs+1):
                mat = np.vstack((mat, find_vectors(i, _num_opt-1)))
            mat = np.hstack((mat, _num_obs - np.sum(mat, axis=1).reshape(-1, 1)))
        return mat

    num_obs, num_opt = sum(obs), len(obs)
    uni = np.full(num_opt, 1/num_opt)
    event_mat = find_vectors(num_obs, num_opt)
    event_prob_h0 = multinomial.pmf(event_mat, num_obs, uni)
    p_obs = multinomial.pmf(obs, num_obs, uni)
    p_val = np.sum(event_prob_h0[event_prob_h0 <= p_obs])
    return p_val

alpha = 0.05
idx_random = defaultdict(set)
for idx, counters in idx2counters.items():
    for key, obs in counters.items():
        p_val = multinom_test(obs)
        if p_val > alpha: idx_random[key].add(idx)

myprint('Step 3: Test Conflicting Knowledge')
alpha = 0.05
def select_alter(obs):
    num_obs, num_opt = sum(obs), len(obs)
    p_hats = (1-obs/num_obs) / (num_opt-1)
    lambdas = 2 * (num_obs*np.log(num_opt) + (num_obs-obs)*np.log(p_hats) + obs*np.log(1-p_hats))
    p_values = chi2.sf(lambdas, df=1)
    
    sig = np.where(p_values < alpha/num_opt)[0]
    valid = np.where(p_hats > 1/num_opt)[0]
    candidates = set(sig) & set(valid)
    alter = [idx for idx in np.argsort(-lambdas) if idx in candidates][0] if len(candidates) else -1
    return alter

idx2alters = defaultdict(lambda: defaultdict(list))
for idx, counters in idx2counters.items():
    for key, counter in counters.items():
        if idx in idx_refusal[key] or idx in idx_random[key]: continue
        
        options = list(enumerate([1,2,3]))
        while len(counter) > 2:
            alter = select_alter(counter)
            if alter == -1: break
            alter_ori, option = options.pop(alter)
            idx2alters[idx][key].append(alter_ori)
            counter = np.delete(counter, alter)

myprint('Step 4: Test No Conflict')
alpha = 0.05
def binom_test(obs):
    num_obs, num_opt = sum(obs), len(obs)    
    p_mode0 = binomtest(k=obs[0], n=num_obs, p=1/num_opt, alternative='greater').pvalue
    p_mode1 = binomtest(k=obs[0], n=num_obs, p=1/num_opt, alternative='less').pvalue
    mode0, mode1 = p_mode0 < alpha/num_opt, p_mode1 < alpha/num_opt
    
    if mode0 and not mode1: return 0
    elif not mode0 and mode1: return 1
    else: return -1
    
idx2modes = defaultdict(dict)
for idx, counters in idx2counters.items():
    for key, counter in counters.items():
        
        options = list(enumerate([1,2,3]))
        if idx in idx_refusal[key] or idx in idx_random[key]: 
            idx2modes[idx][key] = [option[1] for option in options]
            continue
        
        for alter in sorted(idx2alters[idx][key], reverse=True):
            options.pop(alter)
            counter = np.delete(counter, alter)
        if len(options) > 2:
            idx2modes[idx][key] = [option[1] for option in options]
            continue
        
        mode = binom_test(counter)
        idx2modes[idx][key] = [options[mode][1]] if mode != -1 else [option[1] for option in options]
        
json.dump(idx2modes, open(f'{cwd}/Data/Status/{args.dataset}/{args.model.split("/")[-1]}.json', 'w'), indent=4)