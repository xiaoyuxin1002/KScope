import re
import os
import json
import argparse
from tqdm import tqdm
from vllm import LLM, SamplingParams


def get_responses(output):
    responses = {}
    for i, each in enumerate(output):
        match = re.search(pattern, each, re.IGNORECASE | re.DOTALL)
        if match: 
            text = match.group(1).replace('YOUR ANSWER', '').strip()
            if text: responses[i] = text
    return responses

def make_messages(resp):
    INSTRUCTION_PROMPT = \
        """
        Extract the final answer from a response that contains both reasoning and an answer. 
        Return the answer VERBATIM (no paraphrase; preserve original casing, punctuation, spacing). 
        Output ONLY the tags: <answer>…</answer>
        If there is no answer, output exactly: <answer></answer>

        Selection rules (in order):
        1) If the response explicitly marks “Final Answer”, “Answer:”, or similar, copy the text that follows.
        2) Otherwise, pick the most direct, final-looking answer span (e.g., after “Therefore,” “So the result is,” etc.).
        3) If nothing qualifies, return empty tags.

        Do NOT include explanations. Do NOT add or remove characters beyond copying the chosen span.

        Examples:

        Input:
        I think it's 7 because 3+4=7. Final Answer: 7
        Output:
        <answer>7</answer>

        Input:
        Reasoning… So the result is 'Berlin', which matches the clue.
        Output:
        <answer>Berlin</answer>

        Input:
        I'm not sure; need more info.
        Output:
        <answer></answer>

        Input:
        Answer: The quick brown fox.
        Output:
        <answer>The quick brown fox.</answer>
        """
    return [
        {"role": "user", "content": (
            INSTRUCTION_PROMPT + "\n\n"
            f"Input:\n{resp}\n\n"
            f"Output:\n"
        )},
    ]

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Llama-3.1-8B-Instruct')
args = parser.parse_args()
print(f'Extracting {args.model}\'s responses')

num_resp = 100
cwd = os.getcwd()
categories = ['memory', 'context']
pattern = r"<answer>\s*(?!YOUR\s+ANSWER)(.*?)\s*</answer>"
outputs = json.load(open(f'{cwd}/Data/Open/{args.model}_output.json'))

missing_ids, missing_resps = [], []
for idx in tqdm(range(len(outputs))):
    for cat in categories:
        responses = get_responses(outputs[idx][cat])
        for i in range(num_resp):
            if i not in responses:
                missing_ids.append((idx, cat, i))
                missing_resps.append(outputs[idx][cat][i])

sampling_params = SamplingParams(n=1, max_tokens=128, temperature=0.7, seed=0)
judge = LLM(model='google/gemma-2-9b-it', max_model_len=4096, seed=0,
            tensor_parallel_size=2, gpu_memory_utilization=0.9, enable_prefix_caching=True)

batch_size = 100000
for start in range(0, len(missing_resps), batch_size):
    end = min(start + batch_size, len(missing_resps))
    print(f'Processing {start} to {end} / {len(missing_resps)}')
    batch_resps, batch_ids = missing_resps[start:end], missing_ids[start:end]
    batch_messages = [make_messages(resp) for resp in batch_resps]
    batch_chats = judge.chat(batch_messages, sampling_params=sampling_params, use_tqdm=True)
    batch_extractions = [chat.outputs[0].text for chat in batch_chats]
    for (idx, cat, i), ext in zip(batch_ids, batch_extractions):
        outputs[idx][cat][i] += '\n' + ext
json.dump(outputs, open(f'{cwd}/Data/Open/{args.model}_extraction.json', 'w'), indent=4)