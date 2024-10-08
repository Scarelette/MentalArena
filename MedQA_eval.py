import time
import fire
import numpy as np
import random
import jsonlines
import os
# from sklearn.metrics.pairwise import cosine_similarity
from transformers import LlamaTokenizer, LlamaForCausalLM, T5ForConditionalGeneration, T5Tokenizer
cur_tokenizer = None
cur_model = None

def getResponse(prompt, engine='gpt-3.5-turbo', pipeline=None, max_tokens=256):

    msg = []
    msg.append({"role": "system", "content": 'You are a psychiatric expert.'})
    msg.append({"role": "user", "content": prompt})

    if 'llama' in engine.lower() or 'men' in engine.lower():

        outputs = pipeline(
            msg,
            max_new_tokens=256,
            temperature=0.1
        )
        output = outputs[0]["generated_text"][-1]['content']
    
    elif 'gemini' in engine.lower():
        import google.generativeai as genai
        import urllib.request
        import json
        import ssl
        key_pool = ['###', '###']
        
        key = random.choice(key_pool)
        genai.configure(api_key=key)
  
        model = genai.GenerativeModel('gemini-pro')
        # print('ok!!!')

        output = None
        times = 0
        while output==None and times <= 2:
            try:
                times += 1  
                response = model.generate_content(
                    prompt, 
                    safety_settings=[
                        {
                            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            "threshold": "BLOCK_NONE",
                        },
                        {
                            "category": "HARM_CATEGORY_HATE_SPEECH",
                            "threshold": "BLOCK_NONE",
                        },
                        {
                            "category": "HARM_CATEGORY_HARASSMENT",
                            "threshold": "BLOCK_NONE",
                        },
                        {
                            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                            "threshold": "BLOCK_NONE",
                        }
                    ],
                    generation_config=genai.types.GenerationConfig(
                    # Only one candidate for now.
                    max_output_tokens=20,
                    temperature=1.0))
                output = response.text
            except Exception as e:
                output = ''
                print(response.prompt_feedback)
                print('Retrying...')
                time.sleep(60)
        if times >= 2:
            print('Failed! Model Input: ', prompt)
            output = ''
    
    else:
        from openai import OpenAI
        client = OpenAI(api_key="###")

        output = None
        times = 0
        while output is None and times <= 10:
            try:
                times += 1  
                response = client.chat.completions.create(
                    model=engine,
                    max_tokens=256,
                    messages=msg,
                    temperature=0
                    )
                output = response.choices[0].message.content
            except Exception as e:
                print(e)
                print('Retrying...')
                time.sleep(5)
        if times >= 10:
            print('Failed! Model Input: ', prompt)
            output = ''
    
    return output


def getPrompt(question, option):
    prompt = f'As a psychiatrist, {question} Please select one from the following option\nOptions: {option}\nOnly answer option tag'
    return prompt


def run(model='',dataset='PubMedQA', name='test/professional_psychology_test', ours=False):
    score = 0
    total = 0

    if 'llama' in model.lower() or 'men' in model.lower():
        import transformers
        import torch

        model_id = model

        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        # pipeline = None
        # cur_tokenizer = T5Tokenizer.from_pretrained(model_id)
        # cur_model = T5ForConditionalGeneration.from_pretrained(model_id, device_map='auto')

        # cur_tokenizer = LlamaTokenizer.from_pretrained(model_id)
        # cur_model = LlamaForCausalLM.from_pretrained(model_id, device_map='auto')


    if dataset == 'MedQA' or dataset == 'MMLU':
        if dataset == 'MMLU':
            filename = name
        else:
            filename = 'psychosis'
        with jsonlines.open(f'data/{dataset}/psychosis_score_new.jsonl',mode='a') as writer:
            with open(f"data/{dataset}/{filename}.jsonl", "r+", encoding="utf8") as f:
                for item in jsonlines.Reader(f):
                    question = item['question']
                    options = item['options']
                    prompt = f'{question}\nOptions: '
                    for key in options.keys():
                        text = key + '. ' + options[key]
                        prompt += f'\n{text}'
                    prompt += f'\nJust output the correct option without explanation'
                    # prompt += f'\nPlease explain your answer.'
                    if 'llama' in model.lower() or 'men' in model.lower():
                        if '7b' in model:
                            prompt += '\nThe answer should obey the format strictly. For example: \nThe correct option is: A'
                        output = getResponse(prompt, model, pipeline).strip()
                    else:
                        output = getResponse(prompt, model).strip()
                    print('Output: ', output)
                    total += 1
                    if '7b' in model:
                        if ':' in output:
                            index = output.find(':')
                            output = output[index+1:].strip()
                    if '.' in output or len(output) == 1:
                        ans = output[0]
                        if ans == item['answer_idx']:
                            score += 1
                    else:
                        ans = output.strip()
                        if ans in item['answer'] or item['answer'] in ans:
                            score += 1
                    print('Ans: ', ans)

                score = score / total
                score_item = {'model': model, 'name': name, 'score': score}
                print('Score: ', score)
                writer.write(score_item)
    elif dataset == 'MedMCQA':
        if name != 'test/professional_psychology_test':
            score_file = 'data/MedMCQA/illnesses/score.jsonl'
            data_file = f"data/MedMCQA/illnesses/{name}.jsonl"
        else:
            score_file = 'data/MedMCQA/psychosis_score_new.jsonl'
            data_file = 'data/MedMCQA/psychosis.jsonl'
        with jsonlines.open(score_file,mode='a') as writer:
            with open(data_file, "r+", encoding="utf8") as f:
                ans_dict = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
                for item in jsonlines.Reader(f):
                    question = item['question']
                    prompt = f'{question}\nOptions: '
                    
                    prompt += f"\nA. {item['opa']}"
                    prompt += f"\nB. {item['opb']}"
                    prompt += f"\nC. {item['opc']}"
                    prompt += f"\nD. {item['opd']}"

                    prompt += f'\nJust output the correct option without explanation'
                    # prompt += f'\nPlease explain your answer.'
                    if 'llama' in model.lower() or 'men' in model.lower():
                        output = getResponse(prompt, model, pipeline).strip()
                    else:
                        output = getResponse(prompt, model).strip()
                    print(output)
                    total += 1
                    gt = ans_dict[item['cop']]
                    if '.' in output or len(output) == 1:
                        ans = output[0]
                        if ans == gt:
                            score += 1
                    else:
                        ans = output.strip()
                        if ans in gt or gt in ans:
                            score += 1

                score = score / total
                score_item = {'model': model, 'name': name, 'score': score}
                print('Score: ', score)
                writer.write(score_item)

    elif dataset == 'PubMedQA':
        with jsonlines.open('data/PubMedQA/psychosis_score_new.jsonl',mode='a') as writer:
            with open("data/PubMedQA/test_samples.jsonl",'r') as f:
                for item in jsonlines.Reader(f):
                    question = item["QUESTION"]
                    gt = item['final_decision']
                    prompt = question + '\nJust answer with Yes, No or Maybe without explanation'
                    # prompt = question + '\nPlease choose your answer in Yes, No or Maybe and explain your answer.'

                    if 'llama' in model.lower() or 'men' in model.lower():
                        output = getResponse(prompt, model, pipeline).strip()
                    else:
                        output = getResponse(prompt, model).strip()
                    if 'explanation.' in output:
                        index = output.find('explanation.')
                        output = output[index+12:].strip()
                    print(output)

                    total += 1
                    if gt.lower() in output.lower():
                        score += 1
                        if 'yes' in output.lower() and 'no' in output.lower():
                            score -= 1
                score = score / total
                score_item = {'model': model, 'score': score}
                print('Score: ', score)
                writer.write(score_item)

if __name__ == '__main__':
    fire.Fire(run)  
