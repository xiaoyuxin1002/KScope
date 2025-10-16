import re
import os
import json
import argparse
import functools
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from vllm import LLM, SamplingParams
from scipy.stats import binom, binomtest, chi2, multinomial


def make_messages(question, text_a, text_b):
    SYSTEM_PROMPT = \
        """
        You are a precise judge of semantic equivalence.

        Task:
        Given a question Q and two responses A and B, decide if they mean the same thing in the context of Q.

        Guidelines:
        - Judge by **semantic meaning**, not by length or wording.
        - Ignore stylistic or phrasing differences.
        - A brief answer can match a longer one if both convey the same idea.
        - Focus on whether both express the same factual or conceptual content.

        Write one **short** sentence of reasoning, then output exactly a JSON result on a new line:
        {"semantically_equivalent": true|false}
        """
    return [
        {"role": "user", "content": (
            SYSTEM_PROMPT + "\n\n"
            f"Question (shared context):\n{question}\n\n"
            f"Response A:\n{text_a}\n\n"
            f"Response B:\n{text_b}\n\n"
            "Give a **short** one-sentence reasoning, then ONLY the JSON on a new line."
        )},
    ]

def call_judge(idx2messages):
    idx2results = {}
    chats = judge.chat(list(idx2messages.values()), sampling_params=sampling_params, use_tqdm=False)
    responses = [chat.outputs[0].text for chat in chats]
    for (idx, _), response in zip(idx2messages.items(), responses):
        try:
            result = re.search(r"\{.*\}", response, re.DOTALL)
            assert result is not None
            result = json.loads(result.group(0))
            assert 'semantically_equivalent' in result
            assert isinstance(result['semantically_equivalent'], bool)
            idx2results[idx] = result['semantically_equivalent']
        except:
            idx2results[idx] = None
    return idx2results

def test_refusal(counter):
    thres_refusal = binom.isf(alpha, num_resp, 0.5)
    if num_resp - np.sum(counter) >= thres_refusal: return 'Absent'
    return None

def test_random(counter):
    @functools.cache
    def find_vectors(_num_obs, _num_opt):
        mat = _num_obs
        if _num_opt != 1:
            mat = np.zeros((1, _num_opt-1))
            for i in range(1, _num_obs+1):
                mat = np.vstack((mat, find_vectors(i, _num_opt-1)))
            mat = np.hstack((mat, _num_obs - np.sum(mat, axis=1).reshape(-1, 1)))
        return mat
        
    num_obs, num_opt = sum(counter), len(counter)
    uni = np.full(num_opt, 1/num_opt)
    event_mat = find_vectors(num_obs, num_opt)
    event_prob_h0 = multinomial.pmf(event_mat, num_obs, uni)
    p_obs = multinomial.pmf(counter, num_obs, uni)
    p_val = np.sum(event_prob_h0[event_prob_h0 <= p_obs])
    if p_val > alpha: return 'Absent'
    return None

def test_conflict(counter):
    def select_alter(obs):
        obs = np.array(obs)
        num_obs, num_opt = sum(obs), len(obs)
        p_hats = (1-obs/num_obs) / (num_opt-1)
        lambdas = 2 * (num_obs*np.log(num_opt) + (num_obs-obs)*np.log(p_hats) + obs*np.log(1-p_hats))
        p_values = chi2.sf(lambdas, df=1)        
        sig = np.where(p_values < alpha/num_opt)[0]
        valid = np.where(p_hats > 1/num_opt)[0]
        candidates = set(sig) & set(valid)
        alter = [idx for idx in np.argsort(-lambdas) if idx in candidates][0] if len(candidates) and np.any(lambdas != lambdas[0]) else -1
        return alter
    
    alters = []
    options = list(range(len(counter)))
    while len(counter) > 2:
        alter = select_alter(counter)
        if alter == -1: break
        alter_ori = options.pop(alter)
        alters.append(alter_ori)
        counter = np.delete(counter, alter)
    return counter

def test_consistent(counter):
    def binom_test(obs):
        num_obs, num_opt = sum(obs), len(obs)    
        p_mode0 = binomtest(k=obs[0], n=num_obs, p=1/num_opt, alternative='greater').pvalue
        p_mode1 = binomtest(k=obs[0], n=num_obs, p=1/num_opt, alternative='less').pvalue
        mode0, mode1 = p_mode0 < alpha/num_opt, p_mode1 < alpha/num_opt        
        if mode0 and not mode1: return 0
        elif not mode0 and mode1: return 1
        else: return -1

    if len(counter) <= 2:
        mode = binom_test(counter)
        if mode != -1: 
            counter = [counter[mode]]
    return counter

def test_correct(question, answer, clusters, counter):
    mode_clusters = [cluster for cluster in clusters if len(cluster) in counter]
    idx_resp, idx2cluster, idx2messages = 0, {}, {}
    for idx_cluster, cluster in enumerate(mode_clusters):
        for resp in cluster:
            idx2cluster[idx_resp] = idx_cluster
            idx2messages[idx_resp] = make_messages(question, answer, resp)
            idx_resp += 1
    idx2results = call_judge(idx2messages)
    correct_cluster = set([idx2cluster[idx] for idx, result in idx2results.items() if result is True])
    status = 'Conflicting' if len(counter) > 1 else 'Consistent'
    status = status + ' Correct' if len(correct_cluster) else status + ' Wrong'
    return status

def get_status(question, answer, clusters):
    counter = [len(cluster) for cluster in clusters]
    # Step 1: Test Refusal
    status = test_refusal(counter)
    if status: return status
    # Step 2: Test Random Guessing
    status = test_random(counter)
    if status: return status
    # Step 3: Test Conflicting Knowledge
    counter = test_conflict(counter)
    # Step 4: Test No Conflict
    counter = test_consistent(counter)
    status = test_correct(question, answer, clusters, counter)
    return status

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='HotpotQA')
parser.add_argument('--model', type=str, default='Llama-3.1-8B-Instruct')
args = parser.parse_args()

sampling_params = SamplingParams(n=1, max_tokens=128, temperature=0.7, seed=0)
judge = LLM(model='google/gemma-2-9b-it', max_model_len=4096, seed=0,
            tensor_parallel_size=2, gpu_memory_utilization=0.9, enable_prefix_caching=True)

alpha, num_resp = 0.05, 100
thres_refusal = binom.isf(alpha, num_resp, 0.5)

cwd = os.getcwd()
categories = ['memory', 'context']
dataset = pd.read_csv(f'{cwd}/Data/Input/{args.dataset}.csv')
idx2pair = {str(idx):(row['question 1'], row[f'option {row["answer"]}']) for idx, row in dataset.iterrows()}
idx2clusters = json.load(open(f'{cwd}/Data/Open/{args.model}_cluster.json'))

idx2status = defaultdict(dict)
for idx, (question, answer) in tqdm(idx2pair.items()):
    for cat in categories:
        idx2status[idx][cat] = get_status(question, answer, idx2clusters[idx][cat]) if idx in idx2clusters and cat in idx2clusters[idx] else 'Absent'
json.dump(idx2status, open(f'{cwd}/Data/Open/{args.model}_status.json', 'w'), indent=4)