import re
import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
from scipy.stats import binom
from collections import defaultdict
from vllm import LLM, SamplingParams


def get_responses(output):
    responses = {}
    for i, each in enumerate(output):
        match = re.search(pattern, each, re.IGNORECASE | re.DOTALL)
        if match: 
            text = match.group(1).replace('YOUR ANSWER', '').strip()
            if text: responses[i] = text
    return responses

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

def call_judge(indices2messages):
    num_fails = 0
    indices2results = {}
    chats = judge.chat(list(indices2messages.values()), sampling_params=sampling_params, use_tqdm=False)
    responses = [chat.outputs[0].text for chat in chats]
    for (indices, _), response in zip(indices2messages.items(), responses):
        try:
            result = re.search(r"\{.*\}", response, re.DOTALL)
            assert result is not None
            result = json.loads(result.group(0))
            assert 'semantically_equivalent' in result
            assert isinstance(result['semantically_equivalent'], bool)
            indices2results[indices] = result['semantically_equivalent']
        except:
            num_fails += 1
            indices2results[indices] = None
    return indices2results, num_fails

def remove_refusals(question, responses):
    refusal = 'I am unsure or cannot answer the question with the provided information.'
    indices2messages = {(idx,): make_messages(question, text, refusal) for idx, text in responses.items()}
    indices2results, num_fails = call_judge(indices2messages)
    valid_responses = {idx:responses[idx] for (idx,), result in indices2results.items() if not result}
    return valid_responses, num_fails

def string_match(responses):
    matches = defaultdict(list)
    for resp in responses.values():
        matches[resp.lower()].append(resp)
    idx2matches = {i:match for i, match in enumerate(sorted(matches.values(), key=lambda x:len(x[0]), reverse=True))}
    return idx2matches

def get_clusters(question, idx2matches, clusters, num_calls, num_fails):
    if len(idx2matches) == 0: return clusters, num_calls, num_fails
    if len(idx2matches) == 1: clusters.append(idx2matches.popitem()[1]); return clusters, num_calls, num_fails
    _, texts = idx2matches.popitem()
    indices2messages = {i: make_messages(question, texts[0], texts_[0]) for i, texts_ in idx2matches.items()}
    indices2results, num_fails_ = call_judge(indices2messages)
    cluster = texts
    for i, result in indices2results.items():
        if result: cluster += idx2matches.pop(i)
    clusters.append(cluster)
    num_calls += len(indices2messages)
    num_fails += num_fails_
    return get_clusters(question, idx2matches, clusters, num_calls, num_fails)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='HotpotQA')
parser.add_argument('--model', type=str, default='Llama-3.1-8B-Instruct')
args = parser.parse_args()

sampling_params = SamplingParams(n=1, max_tokens=128, temperature=0.7, seed=0)
judge = LLM(model='google/gemma-2-9b-it', max_model_len=4096, seed=0,
            tensor_parallel_size=2, gpu_memory_utilization=0.9, enable_prefix_caching=True)

alpha, num_resp = 0.05, 100
thres_valid = num_resp - binom.isf(alpha, num_resp, 0.5)

cwd = os.getcwd()
categories = ['memory', 'context']
pattern = r"<answer>\s*(?!YOUR\s+ANSWER)(.*?)\s*</answer>"
dataset = pd.read_csv(f'{cwd}/Data/Input/{args.dataset}.csv')
outputs = json.load(open(f'{cwd}/Data/Open/{args.model}_extraction.json'))

idx2responses = {}
for idx, output in enumerate(outputs):
    idx2responses[idx] = {cat:get_responses(output[cat]) for cat in categories}

total_calls, total_fails = 0, 0
idx2clusters = defaultdict(dict)
pbar = tqdm(total=len(idx2responses))
for idx, response in idx2responses.items():

    idx_calls, idx_fails = 0, 0
    question = dataset.iloc[idx]['question 1']
    cat2valid = {cat:'Invalid' for cat in categories}

    for cat in categories:
        if len(response[cat]) > thres_valid:
            idx_calls += len(response[cat])
            response[cat], num_fails = remove_refusals(question, response[cat])
            idx_fails += num_fails

            if len(response[cat]) > thres_valid:
                response[cat] = string_match(response[cat])
                clusters, num_calls, num_fails = get_clusters(question, response[cat], [], 0, 0)
                idx_calls += num_calls
                idx_fails += num_fails
                idx2clusters[idx][cat] = clusters
                cat2valid[cat] = 'Valid'

    total_calls += idx_calls
    total_fails += idx_fails
    pbar.update(1)
    pbar.set_description(f"LLM calls: {idx_calls} / {total_calls} | LLM failures: {idx_fails} / {total_fails} | Memory: {cat2valid['memory']} | Context: {cat2valid['context']}")

print(f'Total LLM calls: {total_calls}')
print(f'Total LLM failures: {total_fails}')
json.dump(idx2clusters, open(f'{cwd}/Data/Open/{args.model}_cluster.json', 'w'), indent=4)