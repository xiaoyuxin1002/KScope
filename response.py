import os
import time
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from vllm import LLM, SamplingParams


system = {
    'Hemonc': 'You are a doctor with a professional medical background.',
    'PubMedQA': 'You are a doctor with a professional medical background.',
    'NQ': 'You are a helpful AI assistant.',
    'HotpotQA': 'You are a helpful AI assistant.'
}
instruction_doc = {
    True: 'You are given some documents and a multiple-choice question.\nBased on the document, select the most appropriate answer from the options provided.',
    False: 'Without relying on any external document, select the most appropriate answer from the options provided.'
}
instruction = 'First, explain your reasoning briefly step-by-step based on the provided information.\nThen, select the most appropriate option and present your response in the required format.'
response = 'Provide your response in the following format:\n<answer>Option [number]</answer>'

# For the open-ended setting:
# instruction_doc = {
#     True: 'You are given some documents and a question.\nBased on the document, provide the most appropriate answer.',
#     False: 'Without relying on any external document, provide the most appropriate answer to the question.'
# }
# instruction = 'First, explain your reasoning briefly step-by-step based on the provided information.\nThen, provide your answer in the required format.'
# response = 'Provide your answer in the following format:\n<answer> YOUR ANSWER </answer>'

def get_message(args, row, qid, if_doc=False):
    prompt = [system[args.dataset]]
    prompt += ['### Instruction:\n' + instruction_doc[if_doc] + instruction]
    if if_doc: prompt += ['### Documents:\n' + row[f'evidence']] # For noisy context: 'retrieved'
    prompt += ['### Question:\n' + row[f'question {qid}']]
    prompt += ['### Choices:\n' + '\n'.join([f'Option {i}: {row[f"option {i}"]}' for i in range(1, 4)])] # For the open-ended setting: remove this line
    prompt += [response]
    message = [{'role':'user', 'content':'\n\n'.join(prompt)}]
    return message

def myprint(text):    
    print(time.strftime("%Y %b %d %a, %H:%M:%S: ", time.localtime()) + text, flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Hemonc')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--num_gpus', type=int, default=1)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    myprint(f'Evaluate {args.model} on {args.dataset}')

    max_len, seed = 4096, 0
    llm = LLM(model=args.model, max_seq_len_to_capture=max_len, seed=seed, 
              tensor_parallel_size=args.num_gpus, gpu_memory_utilization=0.9)

    cwd = os.getcwd()
    dataset = pd.read_csv(f'{cwd}/Data/Input/{args.dataset}.csv')
    num_resp, num_qn = 100, 20
    messages_mem, messages_con = [], []
    for _ in range(num_resp):
        qids = np.random.randint(1, num_qn+1, len(dataset))
        messages_mem.append([get_message(args, row, qids[i], if_doc=False) for i, row in dataset.iterrows()])
        messages_con.append([get_message(args, row, qids[i], if_doc=True) for i, row in dataset.iterrows()])

    responses_mem, responses_con, max_token = [], [], 1024
    for seed in range(num_resp):
        myprint(f'Start predicting with seed = {seed}')
        sampling_params = SamplingParams(n=1, max_tokens=max_token, seed=seed)
        chats_mem = llm.chat(messages_mem[seed], sampling_params, use_tqdm=True)
        chats_con = llm.chat(messages_con[seed], sampling_params, use_tqdm=False)
        responses_mem += [[chat.outputs[0].text for chat in chats_mem]]
        responses_con += [[chat.outputs[0].text for chat in chats_con]]

    responses = []
    for idx in range(len(dataset)):
        response = {'memory':[response[idx] for response in responses_mem], 
                    'context':[response[idx] for response in responses_con]}
        responses.append(response)
    json.dump(responses, open(f'{cwd}/Data/Output/{args.dataset}/{args.model.split("/")[-1]}.json', 'w'), indent=4)

if __name__ == '__main__':
    main()