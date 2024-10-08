import time
import fire
import numpy as np
import random
import jsonlines
import torch
import os
import json
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import transformers
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


def getResponse(engine, prompt, history=None, role=None, sys_prompt=None):

    if history != None:
        msg = history
    else:
        msg = []

    if sys_prompt != None:
        msg.append({"role": "system", "content": sys_prompt})

    msg.append({"role": "user", "content": prompt})

    # print(msg)
    
    if 'llama' in engine:
        outputs = pipeline(
                msg,
                max_new_tokens=256,
                temperature=0.8
            )
        output = outputs[0]["generated_text"][-1]['content']

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
                    temperature=0.8
                    )
                output = response.choices[0].message.content.strip()
            except Exception as e:
                print(e)
                print('Retrying...')
                time.sleep(5)
        if times >= 10:
            print('Failed! Model Input: ', prompt)
            output = ''
    
    if role != None:
        history.append({"role": "system", "content": role + ': ' + output})
        return output, history
    else:
        return output


def try_and_reflect(data, illness_final, history, try_type):
    # try_type: treatment, medication
    patient_state = 0
    times = 0
    max_score = 0
    best_treatment = None
    while times < 7 and max_score < 9:
        # # Doctor treatment
        sys = 'You are a psychiatric expert. Your task is to provide the treatment for the patient.'
        if try_type == 'treatment':
            if best_treatment == None:
                prompt = f'The illness of the patient is: {illness_final} How to treat the patient? Please provide specific treatment. Just answer with one treatment and explain your answer'
            else:
                prompt = f'The illness of the patient is: {illness_final} How to treat the patient? Please provide specific treatment. Just answer with one treatment and explain your answer. Your answer shouldn\'t be the same as previous treatment.'
        else:
            if best_treatment == None:
                prompt = f'The illness of the patient is: {illness_final} What is the proper medicines for the patient? Please provide specific medicines. Just answer with one medicine and explain your answer'
            else:
                prompt = f'The illness of the patient is: {illness_final} What is the proper medicines for the patient? Please provide specific medicines. Just answer with one medicine and explain your answer. Your answer shouldn\'t be the same as previous treatment.'
        if '7b' in model_id:
            treatment, history = getResponse(base_model, sys + '\n' + prompt, history, 'Doctor')
        else:
            treatment, history = getResponse(base_model, prompt, history, 'Doctor', sys)
        print(f'{try_type}: ', treatment)
    
        # # # Patient ensure
        sys = f"Imagine you are {data['name']}, a patient who has been experiencing mental health challenges. You have been attending therapy sessions for several weeks. Align your responses with {data['name']}'s background information provided in the 'Relevant history' section. Your thought process should be guided by the cognitive conceptualization diagram in the 'Cognitive Conceptualization Diagram' section, but avoid directly referencing the diagram as a real patient would not explicitly think in those terms. \n\nPatient History: {data['history']}\n\nCognitive Conceptualization Diagram:\nIntermediate Beliefs: {data['intermediate_belief']}\nIntermediate Beliefs during Depression: {data['intermediate_belief_depression']}\nCoping Strategies: {data['coping_strategies']}\n\nYou will be asked about your experiences over the past week. Engage in a conversation with the therapist regarding the following situation and behavior. Use the provided emotions and automatic thoughts as a reference, but do not disclose the cognitive conceptualization diagram directly. Instead, allow your responses to be informed by the diagram, enabling the therapist to infer your thought processes.\n\nSituation: {data['situation']}\nAutomatic Thoughts: {data['auto_thought']}\nEmotions: {data['emotion']}\nBehavior: {data['behavior']}\n\nIn the upcoming conversation, you will simulate {data['name']} during the therapy session, while the user will play the role of the therapist. Adhere to the following guidelines:\n1. Emulate the demeanor and responses of a genuine patient to ensure authenticity in your interactions. Use natural language, including hesitations, pauses, and emotional expressions, to enhance the realism of your responses.\n2. Gradually reveal deeper concerns and core issues, as a real patient often requires extensive dialogue before delving into more sensitive topics. This gradual revelation creates challenges for therapists in identifying the patient's true thoughts and emotions.\n3. Maintain consistency with {data['name']}'s profile throughout the conversation. Ensure that your responses align with the provided background information, cognitive conceptualization diagram, and the specific situation, thoughts, emotions, and behaviors described.\n4. Engage in a dynamic and interactive conversation with the therapist. Respond to their questions and prompts in a way that feels authentic and true to {data['name']}'s character. Allow the conversation to flow naturally, and avoid providing abrupt or disconnected responses.\n\nYou are now {data['name']}. Respond to the therapist's prompts as {data['name']} would, regardless of the specific questions asked. Limit each of your responses to a maximum of 5 sentences. If the therapist begins the conversation with a greeting like \"Hi,\" initiate the conversation as the patient."
        if try_type == 'treatment':
            prompt = f'What may be happened on your healthy state after the treatment? Treatment: {treatment}'
        else:
            prompt = f'What may be happened on your healthy state after taking the medicine? Medicine: {treatment}'
        if '7b' in model_id:
            patient_sure, cur_his = getResponse(base_model, sys + '\n' + prompt, [], 'Patient')
        else:
            patient_sure, cur_his = getResponse(base_model, prompt, [], 'Patient', sys)
        print('Patient: ', patient_sure)
        sys = f"Imagine you are {data['name']}, a patient who has been experiencing mental health challenges. You have been attending therapy sessions for several weeks. Align your responses with {data['name']}'s background information provided in the 'Relevant history' section. Your thought process should be guided by the cognitive conceptualization diagram in the 'Cognitive Conceptualization Diagram' section, but avoid directly referencing the diagram as a real patient would not explicitly think in those terms. \n\nPatient History: {data['history']}\n\nCognitive Conceptualization Diagram:\nIntermediate Beliefs: {data['intermediate_belief']}\nIntermediate Beliefs during Depression: {data['intermediate_belief_depression']}\nCoping Strategies: {data['coping_strategies']}\n\nYou will be asked about your experiences over the past week. Engage in a conversation with the therapist regarding the following situation and behavior. Use the provided emotions and automatic thoughts as a reference, but do not disclose the cognitive conceptualization diagram directly. Instead, allow your responses to be informed by the diagram, enabling the therapist to infer your thought processes.\n\nSituation: {data['situation']}\nAutomatic Thoughts: {data['auto_thought']}\nEmotions: {data['emotion']}\nBehavior: {data['behavior']}\n\nIn the upcoming conversation, you will simulate {data['name']} during the therapy session, while the user will play the role of the therapist. Adhere to the following guidelines:\n1. Emulate the demeanor and responses of a genuine patient to ensure authenticity in your interactions. Use natural language, including hesitations, pauses, and emotional expressions, to enhance the realism of your responses.\n2. Gradually reveal deeper concerns and core issues, as a real patient often requires extensive dialogue before delving into more sensitive topics. This gradual revelation creates challenges for therapists in identifying the patient's true thoughts and emotions.\n3. Maintain consistency with {data['name']}'s profile throughout the conversation. Ensure that your responses align with the provided background information, cognitive conceptualization diagram, and the specific situation, thoughts, emotions, and behaviors described.\n4. Engage in a dynamic and interactive conversation with the therapist. Respond to their questions and prompts in a way that feels authentic and true to {data['name']}'s character. Allow the conversation to flow naturally, and avoid providing abrupt or disconnected responses.\n\nYou are now {data['name']}. Respond to the therapist's prompts as {data['name']} would, regardless of the specific questions asked. Limit each of your responses to a maximum of 5 sentences. If the therapist begins the conversation with a greeting like \"Hi,\" initiate the conversation as the patient."
        prompt = f'After treatment, your healthy state is: {patient_sure}\nPlease give a score between 1 to 10 for your healthy state. 1-bad, 10-good. Just answer without explanation.'
        patient_state, cur_his = getResponse('gpt-4-turbo-2024-04-09', prompt, [], 'Patient', sys)
        try:
            patient_state = int(patient_state)
        except:
            import re
            score_list = re.findall(r"\d+",patient_state)
            if len(score_list) >= 1:
                score_str = score_list[0]
                patient_state = int(score_str)
            else:
                patient_state = 0

        print('Score: ', patient_state)

        times += 1
        if patient_state > max_score:
            max_score = patient_state
            best_treatment = treatment

    print(f'Best {try_type}: ', best_treatment)
    print('Max score: ', max_score)

    return best_treatment, max_score


def calculate_similarity(sentence1, sentence2):
    def get_sentence_embedding(sentence):
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            sentence_embedding = torch.mean(outputs.last_hidden_state, dim=1)
        return sentence_embedding
    embedding1 = get_sentence_embedding(sentence1)
    embedding2 = get_sentence_embedding(sentence2)
    similarity = cosine_similarity(embedding1, embedding2)
    return similarity


def find_proper_principle(coping_strategies, principle_dict):
    import re

    score_dict = dict()
    for key in principle_dict:
        principle_list = principle_dict[key]
        sim_final = 0
        for p in principle_list:
            sim_cur = calculate_similarity(p, coping_strategies)
            if sim_cur > sim_final:
                sim_final = sim_cur
        score_dict[key] = sim_final
    
    sorted_dict_desc = dict(sorted(score_dict.items(), key=lambda item: item[1], reverse=True)[:5])

    prompt = f'The coping strategies for the patient is: {coping_strategies}\nThere are several groups of principles. Please pick the most appropriate principles for the patient.\n'
    num = 1
    new_dict = dict()
    for key in sorted_dict_desc:
        principle_list = principle_dict[key]
        cur_p = f'Group {num}: '
        for principle in principle_list:
            cur_p += principle
            cur_p += ' '
        cur_p += '\n'
        prompt += cur_p
        new_dict[str(num)] = principle_list 
        num += 1

    prompt += 'Please output the number of the most appropriate group without explanation.'
    # print('Prompt: ', prompt)
    output, cur_his = getResponse('gpt-4-turbo-2024-04-09', prompt, [], 'Patient')

    number = re.findall(r'\d+', output)[0]
    if number in new_dict.keys():
        best_principle = new_dict[number]
    else:
        if ':' in output:
            index = output.find(':')
            output = output[index+1:]
        output = output.strip()
        best_principle = [output]

    # print('Output: ', best_principle)

    return best_principle


def model_decode(history, principle_list, cognitive_data):
    # brain: 
    # Intermediate Beliefs: {data['intermediate_belief']}\nIntermediate Beliefs during Depression: {data['intermediate_belief_depression']}\nCoping Strategies: {data['coping_strategies']}
    # Automatic Thoughts: {data['auto_thought']}\nEmotions: {data['emotion']}\nBehavior: {data['behavior']}
    # principles: predict some given others
    # stop when semantic similar
    # if not similar, get feedback and continue
    feedback = ''
    prompt_b_1 = 'What are the Intermediate Beliefs of the patient?'
    prompt_b_2 = 'What are the Intermediate Beliefs during Depression of the patient?'
    prompt_b_3 = 'What are the Coping Strategies of the patient?'
    prompt_b_4 = f"In the situation: {cognitive_data['situation']}\nHow about the patient's thought, emotion and behaviors in the situation above?"
    prompt_behavior = 'Please analysis the behavior principles for the patient'
    
    output_b_1, history = getResponse(base_model, prompt_b_1, history, 'Doctor')
    output_b_2, history = getResponse(base_model, prompt_b_2, history, 'Doctor')
    output_b_3, history = getResponse(base_model, prompt_b_3, history, 'Doctor')
    output_b_4, history = getResponse(base_model, prompt_b_4, history, 'Doctor')
    output_behavior, history = getResponse(base_model, prompt_behavior, history, 'Doctor')

    gt_b_1 = cognitive_data['intermediate_belief']
    gt_b_2 = cognitive_data['intermediate_belief_depression']
    gt_b_3 = cognitive_data['coping_strategies']
    gt_b_4 = f"Automatic Thoughts: {cognitive_data['auto_thought']}\nEmotions: {cognitive_data['emotion']}\nBehavior: {cognitive_data['behavior']}"
    gt_behavior = ''
    for p in principle_list:
        gt_behavior += p
        gt_behavior += ' '

    s_b_1 = calculate_similarity(output_b_1, gt_b_1)
    s_b_2 = calculate_similarity(output_b_2, gt_b_2)
    s_b_3 = calculate_similarity(output_b_3, gt_b_3)
    s_b_4 = calculate_similarity(output_b_4, gt_b_4)
    s_behavior = calculate_similarity(output_behavior, gt_behavior)

    brain_dict = {'intermediate_belief': s_b_1, 'intermediate_belief_depression': s_b_2, 'coping_strategies': s_b_3, 'situation': s_b_4}
    brain_gt_dict = {'intermediate_belief': gt_b_1, 'intermediate_belief_depression': gt_b_2, 'coping_strategies': gt_b_3, 'situation': gt_b_4}
    brain_output_dict = {'intermediate_belief': output_b_1, 'intermediate_belief_depression': output_b_2, 'coping_strategies': output_b_3, 'situation': output_b_4}
    tag_brain = 0
    brain_gt_str = ''
    brain_output_str = ''
    for key in brain_dict.keys():
        score = brain_dict[key]
        if score < 0.9:
            tag_brain = 1
            cur_gt = brain_gt_dict[key]
            cur_output = brain_output_dict[key]
            brain_gt_str += f'\n{key}:\n{cur_gt}'
            brain_output_str += f'\n{key}:\n{cur_output}'

    if tag_brain == 1:
        prompt = f'The cognitive model of the mental health patient is:\n{brain_gt_str}\nThe diagnose of the doctor is:\n{brain_output_str}\n\nWhat can the doctor ask the patient to diagnose accurately?'
        output, cur_his = getResponse('gpt-4-turbo-2024-04-09', prompt, [], 'Patient')
        feedback += output
    
    if s_behavior < 0.9:
        prompt = f'The behavior principles of the mental health patient is:\n{gt_behavior}\nThe diagnose of the doctor is:\n{output_behavior}\n\nWhat can the doctor ask the patient to diagnose accurately?'
        output, cur_his = getResponse('gpt-4-turbo-2024-04-09', prompt, [], 'Patient')
        feedback += output

    return feedback


def run(base_model, output_file):

    model_id = base_model

    if 'llama' in base_model:
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
    else:
        pipeline = None

    with open("data/Patient-Psi-CM_dataset.json", "r") as f:
        with open("data/roleplay.json", "r") as f_p:
            with jsonlines.open(output_file, mode='a') as writer:
                content = json.load(f)
                principle_dict = json.load(f_p)

                for index in range(len(content)):
                    data = content[index]
                    print(data['name'])

                    principle_list = find_proper_principle(data['coping_strategies'], principle_dict)
                    prompt_p = 'Your statement should obey the following principles:'
                    for p in principle_list:
                        prompt_p += f'\n{p}'

                    history = []
                    prompt = f"You are {data['name']}, a patient who has been experiencing mental health challenges. You have been attending therapy sessions for several weeks. Your task is to engage in a conversation with the therapist as {data['name']} would during a cognitive behavioral therapy (CBT) session. Align your responses with {data['name']}'s background information provided in the 'Relevant history' section. Your thought process should be guided by the cognitive conceptualization diagram in the 'Cognitive Conceptualization Diagram' section, but avoid directly referencing the diagram as a real patient would not explicitly think in those terms. \n\nPatient History: {data['history']}\n\nCognitive Conceptualization Diagram:\nIntermediate Beliefs: {data['intermediate_belief']}\nIntermediate Beliefs during Depression: {data['intermediate_belief_depression']}\nCoping Strategies: {data['coping_strategies']}\n\nYou will be asked about your experiences over the past week. Engage in a conversation with the therapist regarding the following situation and behavior. Use the provided emotions and automatic thoughts as a reference, but do not disclose the cognitive conceptualization diagram directly. Instead, allow your responses to be informed by the diagram, enabling the therapist to infer your thought processes.\n\nSituation: {data['situation']}\nAutomatic Thoughts: {data['auto_thought']}\nEmotions: {data['emotion']}\nBehavior: {data['behavior']}\n\nIn the upcoming conversation, you will simulate {data['name']} during the therapy session, while the user will play the role of the therapist. Adhere to the following guidelines:\n1. Emulate the demeanor and responses of a genuine patient to ensure authenticity in your interactions. Use natural language, including hesitations, pauses, and emotional expressions, to enhance the realism of your responses.\n2. Gradually reveal deeper concerns and core issues, as a real patient often requires extensive dialogue before delving into more sensitive topics. This gradual revelation creates challenges for therapists in identifying the patient's true thoughts and emotions.\n3. Maintain consistency with {data['name']}'s profile throughout the conversation. Ensure that your responses align with the provided background information, cognitive conceptualization diagram, and the specific situation, thoughts, emotions, and behaviors described.\n4. Engage in a dynamic and interactive conversation with the therapist. Respond to their questions and prompts in a way that feels authentic and true to {data['name']}'s character. Allow the conversation to flow naturally, and avoid providing abrupt or disconnected responses.\n\nYou are now {data['name']}. Respond to the therapist's prompts as {data['name']} would, regardless of the specific questions asked. Limit each of your responses to a maximum of 5 sentences. If the therapist begins the conversation with a greeting like \"Hi,\" initiate the conversation as the patient."
                    prompt += f'\n{prompt_p}'
                    output, history = getResponse(base_model, prompt, history, 'Patient')
                    print('Patient: ', output)

                    # 1
                    sys = 'You are a psychiatric expert. You try to help a mental patient solve her/his problem. Your task is to figure out What kind of mental illness the patient has and the severity of the illness. You can ask for patient\'s personal infomation, specific information on the symptom(emotional, cognitive, behavior, psysiological), and the reason behind that(relevent history event). You can also ask other questions which could help you to diagnose disease.'
                    if '7b' in model_id:
                        output, history = getResponse(base_model, sys + '\nPlease continue the conversation with the patient', history, 'Doctor')
                    else:
                        output, history = getResponse(base_model, 'Please continue the conversation with the patient', history, 'Doctor', sys)
                    print('Doctor: ', output)
                    sys = f"You are {data['name']}, a patient who has been experiencing mental health challenges. You have been attending therapy sessions for several weeks. Your task is to engage in a conversation with the therapist as {data['name']} would during a cognitive behavioral therapy (CBT) session. Align your responses with {data['name']}'s background information provided in the 'Relevant history' section. Your thought process should be guided by the cognitive conceptualization diagram in the 'Cognitive Conceptualization Diagram' section, but avoid directly referencing the diagram as a real patient would not explicitly think in those terms. \n\nPatient History: {data['history']}\n\nCognitive Conceptualization Diagram:\nIntermediate Beliefs: {data['intermediate_belief']}\nIntermediate Beliefs during Depression: {data['intermediate_belief_depression']}\nCoping Strategies: {data['coping_strategies']}\n\nYou will be asked about your experiences over the past week. Engage in a conversation with the therapist regarding the following situation and behavior. Use the provided emotions and automatic thoughts as a reference, but do not disclose the cognitive conceptualization diagram directly. Instead, allow your responses to be informed by the diagram, enabling the therapist to infer your thought processes.\n\nSituation: {data['situation']}\nAutomatic Thoughts: {data['auto_thought']}\nEmotions: {data['emotion']}\nBehavior: {data['behavior']}\n\nIn the upcoming conversation, you will simulate {data['name']} during the therapy session, while the user will play the role of the therapist. Adhere to the following guidelines:\n1. Emulate the demeanor and responses of a genuine patient to ensure authenticity in your interactions. Use natural language, including hesitations, pauses, and emotional expressions, to enhance the realism of your responses.\n2. Gradually reveal deeper concerns and core issues, as a real patient often requires extensive dialogue before delving into more sensitive topics. This gradual revelation creates challenges for therapists in identifying the patient's true thoughts and emotions.\n3. Maintain consistency with {data['name']}'s profile throughout the conversation. Ensure that your responses align with the provided background information, cognitive conceptualization diagram, and the specific situation, thoughts, emotions, and behaviors described.\n4. Engage in a dynamic and interactive conversation with the therapist. Respond to their questions and prompts in a way that feels authentic and true to {data['name']}'s character. Allow the conversation to flow naturally, and avoid providing abrupt or disconnected responses.\n\nYou are now {data['name']}. Respond to the therapist's prompts as {data['name']} would, regardless of the specific questions asked. Limit each of your responses to a maximum of 5 sentences. If the therapist begins the conversation with a greeting like \"Hi,\" initiate the conversation as the patient."
                    sys += f'\n{prompt_p}'
                    output, history = getResponse(base_model, sys, history, 'Patient')
                    print('Patient: ', output)

                    # 2
                    sys = 'You are a psychiatric expert. You try to help a mental patient solve her/his problem. Your task is to figure out What kind of mental illness the patient has and the severity of the illness. You can ask for patient\'s personal infomation, specific information on the symptom(emotional, cognitive, behavior, psysiological), and the reason behind that(relevent history event). You can also ask other questions which could help you to diagnose disease.'
                    if '7b' in model_id:
                        output, history = getResponse(base_model, sys + '\nPlease continue the conversation with the patient', history, 'Doctor')
                    else:
                        output, history = getResponse(base_model, 'Please continue the conversation with the patient', history, 'Doctor', sys)
                    print('Doctor: ', output)
                    sys = f"You are {data['name']}, a patient who has been experiencing mental health challenges. You have been attending therapy sessions for several weeks. Your task is to engage in a conversation with the therapist as {data['name']} would during a cognitive behavioral therapy (CBT) session. Align your responses with {data['name']}'s background information provided in the 'Relevant history' section. Your thought process should be guided by the cognitive conceptualization diagram in the 'Cognitive Conceptualization Diagram' section, but avoid directly referencing the diagram as a real patient would not explicitly think in those terms. \n\nPatient History: {data['history']}\n\nCognitive Conceptualization Diagram:\nIntermediate Beliefs: {data['intermediate_belief']}\nIntermediate Beliefs during Depression: {data['intermediate_belief_depression']}\nCoping Strategies: {data['coping_strategies']}\n\nYou will be asked about your experiences over the past week. Engage in a conversation with the therapist regarding the following situation and behavior. Use the provided emotions and automatic thoughts as a reference, but do not disclose the cognitive conceptualization diagram directly. Instead, allow your responses to be informed by the diagram, enabling the therapist to infer your thought processes.\n\nSituation: {data['situation']}\nAutomatic Thoughts: {data['auto_thought']}\nEmotions: {data['emotion']}\nBehavior: {data['behavior']}\n\nIn the upcoming conversation, you will simulate {data['name']} during the therapy session, while the user will play the role of the therapist. Adhere to the following guidelines:\n1. Emulate the demeanor and responses of a genuine patient to ensure authenticity in your interactions. Use natural language, including hesitations, pauses, and emotional expressions, to enhance the realism of your responses.\n2. Gradually reveal deeper concerns and core issues, as a real patient often requires extensive dialogue before delving into more sensitive topics. This gradual revelation creates challenges for therapists in identifying the patient's true thoughts and emotions.\n3. Maintain consistency with {data['name']}'s profile throughout the conversation. Ensure that your responses align with the provided background information, cognitive conceptualization diagram, and the specific situation, thoughts, emotions, and behaviors described.\n4. Engage in a dynamic and interactive conversation with the therapist. Respond to their questions and prompts in a way that feels authentic and true to {data['name']}'s character. Allow the conversation to flow naturally, and avoid providing abrupt or disconnected responses.\n\nYou are now {data['name']}. Respond to the therapist's prompts as {data['name']} would, regardless of the specific questions asked. Limit each of your responses to a maximum of 5 sentences. If the therapist begins the conversation with a greeting like \"Hi,\" initiate the conversation as the patient."
                    sys += f'\n{prompt_p}'
                    output, history = getResponse(base_model, sys, history, 'Patient')
                    print('Patient: ', output)

                    # 3
                    sys = 'You are a psychiatric expert. You try to help a mental patient solve her/his problem. Your task is to figure out What kind of mental illness the patient has and the severity of the illness. You can ask for patient\'s personal infomation, specific information on the symptom(emotional, cognitive, behavior, psysiological), and the reason behind that(relevent history event). You can also ask other questions which could help you to diagnose disease.'
                    if '7b' in model_id:
                        output, history = getResponse(base_model, sys + '\nPlease continue the conversation with the patient', history, 'Doctor')
                    else:
                        output, history = getResponse(base_model, 'Please continue the conversation with the patient', history, 'Doctor', sys)
                    print('Doctor: ', output)
                    sys = f"You are {data['name']}, a patient who has been experiencing mental health challenges. You have been attending therapy sessions for several weeks. Your task is to engage in a conversation with the therapist as {data['name']} would during a cognitive behavioral therapy (CBT) session. Align your responses with {data['name']}'s background information provided in the 'Relevant history' section. Your thought process should be guided by the cognitive conceptualization diagram in the 'Cognitive Conceptualization Diagram' section, but avoid directly referencing the diagram as a real patient would not explicitly think in those terms. \n\nPatient History: {data['history']}\n\nCognitive Conceptualization Diagram:\nIntermediate Beliefs: {data['intermediate_belief']}\nIntermediate Beliefs during Depression: {data['intermediate_belief_depression']}\nCoping Strategies: {data['coping_strategies']}\n\nYou will be asked about your experiences over the past week. Engage in a conversation with the therapist regarding the following situation and behavior. Use the provided emotions and automatic thoughts as a reference, but do not disclose the cognitive conceptualization diagram directly. Instead, allow your responses to be informed by the diagram, enabling the therapist to infer your thought processes.\n\nSituation: {data['situation']}\nAutomatic Thoughts: {data['auto_thought']}\nEmotions: {data['emotion']}\nBehavior: {data['behavior']}\n\nIn the upcoming conversation, you will simulate {data['name']} during the therapy session, while the user will play the role of the therapist. Adhere to the following guidelines:\n1. Emulate the demeanor and responses of a genuine patient to ensure authenticity in your interactions. Use natural language, including hesitations, pauses, and emotional expressions, to enhance the realism of your responses.\n2. Gradually reveal deeper concerns and core issues, as a real patient often requires extensive dialogue before delving into more sensitive topics. This gradual revelation creates challenges for therapists in identifying the patient's true thoughts and emotions.\n3. Maintain consistency with {data['name']}'s profile throughout the conversation. Ensure that your responses align with the provided background information, cognitive conceptualization diagram, and the specific situation, thoughts, emotions, and behaviors described.\n4. Engage in a dynamic and interactive conversation with the therapist. Respond to their questions and prompts in a way that feels authentic and true to {data['name']}'s character. Allow the conversation to flow naturally, and avoid providing abrupt or disconnected responses.\n\nYou are now {data['name']}. Respond to the therapist's prompts as {data['name']} would, regardless of the specific questions asked. Limit each of your responses to a maximum of 5 sentences. If the therapist begins the conversation with a greeting like \"Hi,\" initiate the conversation as the patient."
                    sys += f'\n{prompt_p}'
                    output, history = getResponse(base_model, sys, history, 'Patient')
                    print('Patient: ', output)

                    # 4
                    sys = 'You are a psychiatric expert. You try to help a mental patient solve her/his problem. Your task is to figure out What kind of mental illness the patient has and the severity of the illness. You can ask for patient\'s personal infomation, specific information on the symptom(emotional, cognitive, behavior, psysiological), and the reason behind that(relevent history event). You can also ask other questions which could help you to diagnose disease.'
                    if '7b' in model_id:
                        output, history = getResponse(base_model, sys + '\nPlease continue the conversation with the patient', history, 'Doctor')
                    else:
                        output, history = getResponse(base_model, 'Please continue the conversation with the patient', history, 'Doctor', sys)
                    print('Doctor: ', output)
                    sys = f"You are {data['name']}, a patient who has been experiencing mental health challenges. You have been attending therapy sessions for several weeks. Your task is to engage in a conversation with the therapist as {data['name']} would during a cognitive behavioral therapy (CBT) session. Align your responses with {data['name']}'s background information provided in the 'Relevant history' section. Your thought process should be guided by the cognitive conceptualization diagram in the 'Cognitive Conceptualization Diagram' section, but avoid directly referencing the diagram as a real patient would not explicitly think in those terms. \n\nPatient History: {data['history']}\n\nCognitive Conceptualization Diagram:\nIntermediate Beliefs: {data['intermediate_belief']}\nIntermediate Beliefs during Depression: {data['intermediate_belief_depression']}\nCoping Strategies: {data['coping_strategies']}\n\nYou will be asked about your experiences over the past week. Engage in a conversation with the therapist regarding the following situation and behavior. Use the provided emotions and automatic thoughts as a reference, but do not disclose the cognitive conceptualization diagram directly. Instead, allow your responses to be informed by the diagram, enabling the therapist to infer your thought processes.\n\nSituation: {data['situation']}\nAutomatic Thoughts: {data['auto_thought']}\nEmotions: {data['emotion']}\nBehavior: {data['behavior']}\n\nIn the upcoming conversation, you will simulate {data['name']} during the therapy session, while the user will play the role of the therapist. Adhere to the following guidelines:\n1. Emulate the demeanor and responses of a genuine patient to ensure authenticity in your interactions. Use natural language, including hesitations, pauses, and emotional expressions, to enhance the realism of your responses.\n2. Gradually reveal deeper concerns and core issues, as a real patient often requires extensive dialogue before delving into more sensitive topics. This gradual revelation creates challenges for therapists in identifying the patient's true thoughts and emotions.\n3. Maintain consistency with {data['name']}'s profile throughout the conversation. Ensure that your responses align with the provided background information, cognitive conceptualization diagram, and the specific situation, thoughts, emotions, and behaviors described.\n4. Engage in a dynamic and interactive conversation with the therapist. Respond to their questions and prompts in a way that feels authentic and true to {data['name']}'s character. Allow the conversation to flow naturally, and avoid providing abrupt or disconnected responses.\n\nYou are now {data['name']}. Respond to the therapist's prompts as {data['name']} would, regardless of the specific questions asked. Limit each of your responses to a maximum of 5 sentences. If the therapist begins the conversation with a greeting like \"Hi,\" initiate the conversation as the patient."
                    sys += f'\n{prompt_p}'
                    output, history = getResponse(base_model, sys, history, 'Patient')
                    print('Patient: ', output)

                    # Symptom decode
                    times = 0
                    feedback = model_decode(history, principle_list, data)
                    while len(feedback) > 0 and times < 5:
                        sys = 'You are a psychiatric expert. You try to help a mental patient solve her/his problem. Your task is to figure out What kind of mental illness the patient has and the severity of the illness. You can ask for patient\'s personal infomation, specific information on the symptom(emotional, cognitive, behavior, psysiological), and the reason behind that(relevent history event). You can also ask other questions which could help you to diagnose disease.'
                        sys + f'\nInstruct: {feedback}'
                        if '7b' in model_id:
                            output, history = getResponse(base_model, sys + '\nPlease continue the conversation with the patient', history, 'Doctor')
                        else:
                            output, history = getResponse(base_model, 'Please continue the conversation with the patient', history, 'Doctor', sys)
                        print('Doctor: ', output)
                        sys = f"You are {data['name']}, a patient who has been experiencing mental health challenges. You have been attending therapy sessions for several weeks. Your task is to engage in a conversation with the therapist as {data['name']} would during a cognitive behavioral therapy (CBT) session. Align your responses with {data['name']}'s background information provided in the 'Relevant history' section. Your thought process should be guided by the cognitive conceptualization diagram in the 'Cognitive Conceptualization Diagram' section, but avoid directly referencing the diagram as a real patient would not explicitly think in those terms. \n\nPatient History: {data['history']}\n\nCognitive Conceptualization Diagram:\nIntermediate Beliefs: {data['intermediate_belief']}\nIntermediate Beliefs during Depression: {data['intermediate_belief_depression']}\nCoping Strategies: {data['coping_strategies']}\n\nYou will be asked about your experiences over the past week. Engage in a conversation with the therapist regarding the following situation and behavior. Use the provided emotions and automatic thoughts as a reference, but do not disclose the cognitive conceptualization diagram directly. Instead, allow your responses to be informed by the diagram, enabling the therapist to infer your thought processes.\n\nSituation: {data['situation']}\nAutomatic Thoughts: {data['auto_thought']}\nEmotions: {data['emotion']}\nBehavior: {data['behavior']}\n\nIn the upcoming conversation, you will simulate {data['name']} during the therapy session, while the user will play the role of the therapist. Adhere to the following guidelines:\n1. Emulate the demeanor and responses of a genuine patient to ensure authenticity in your interactions. Use natural language, including hesitations, pauses, and emotional expressions, to enhance the realism of your responses.\n2. Gradually reveal deeper concerns and core issues, as a real patient often requires extensive dialogue before delving into more sensitive topics. This gradual revelation creates challenges for therapists in identifying the patient's true thoughts and emotions.\n3. Maintain consistency with {data['name']}'s profile throughout the conversation. Ensure that your responses align with the provided background information, cognitive conceptualization diagram, and the specific situation, thoughts, emotions, and behaviors described.\n4. Engage in a dynamic and interactive conversation with the therapist. Respond to their questions and prompts in a way that feels authentic and true to {data['name']}'s character. Allow the conversation to flow naturally, and avoid providing abrupt or disconnected responses.\n\nYou are now {data['name']}. Respond to the therapist's prompts as {data['name']} would, regardless of the specific questions asked. Limit each of your responses to a maximum of 5 sentences. If the therapist begins the conversation with a greeting like \"Hi,\" initiate the conversation as the patient."
                        sys += f'\n{prompt_p}'
                        output, history = getResponse(base_model, sys, history, 'Patient')
                        print('Patient: ', output)
                        feedback = model_decode(history, principle_list, data)
                        times += 1


                    # Doctor diagnose
                    sys = 'You are a psychiatric expert. Your task is to diagnose for the patient.'
                    if '7b' in model_id:
                        diagnose_1, history = getResponse(base_model, sys + '\nWhat is the likely diagnosis of the patient? Just answer with one illness and explain your answer', history, 'Doctor')
                    else:
                        diagnose_1, history = getResponse(base_model, 'What is the likely diagnosis of the patient? Just answer with one illness and explain your answer', history, 'Doctor', sys)
                    print('Diagnose 1: ', diagnose_1)

                    sys = 'You are a psychiatric expert. Your task is to diagnose for the patient.'
                    if '7b' in model_id:
                        diagnose_2, history = getResponse(base_model, sys + '\nWhat is the likely diagnosis of the patient? Just answer with one illness and explain your answer. Your answer shouldn\'t be the same as previous diagnose', history, 'Doctor')
                    else:
                        diagnose_2, history = getResponse(base_model, 'What is the likely diagnosis of the patient? Just answer with one illness and explain your answer. Your answer shouldn\'t be the same as previous diagnose', history, 'Doctor', sys)
                    print('Diagnose 2: ', diagnose_2)

                    # # # Patient ensure
                    sys = f"Imagine you are {data['name']}, a patient who has been experiencing mental health challenges. You have been attending therapy sessions for several weeks. Align your responses with {data['name']}'s background information provided in the 'Relevant history' section. Your thought process should be guided by the cognitive conceptualization diagram in the 'Cognitive Conceptualization Diagram' section, but avoid directly referencing the diagram as a real patient would not explicitly think in those terms. \n\nPatient History: {data['history']}\n\nCognitive Conceptualization Diagram:\nIntermediate Beliefs: {data['intermediate_belief']}\nIntermediate Beliefs during Depression: {data['intermediate_belief_depression']}\nCoping Strategies: {data['coping_strategies']}\n\nYou will be asked about your experiences over the past week. Engage in a conversation with the therapist regarding the following situation and behavior. Use the provided emotions and automatic thoughts as a reference, but do not disclose the cognitive conceptualization diagram directly. Instead, allow your responses to be informed by the diagram, enabling the therapist to infer your thought processes.\n\nSituation: {data['situation']}\nAutomatic Thoughts: {data['auto_thought']}\nEmotions: {data['emotion']}\nBehavior: {data['behavior']}\n\nIn the upcoming conversation, you will simulate {data['name']} during the therapy session, while the user will play the role of the therapist. Adhere to the following guidelines:\n1. Emulate the demeanor and responses of a genuine patient to ensure authenticity in your interactions. Use natural language, including hesitations, pauses, and emotional expressions, to enhance the realism of your responses.\n2. Gradually reveal deeper concerns and core issues, as a real patient often requires extensive dialogue before delving into more sensitive topics. This gradual revelation creates challenges for therapists in identifying the patient's true thoughts and emotions.\n3. Maintain consistency with {data['name']}'s profile throughout the conversation. Ensure that your responses align with the provided background information, cognitive conceptualization diagram, and the specific situation, thoughts, emotions, and behaviors described.\n4. Engage in a dynamic and interactive conversation with the therapist. Respond to their questions and prompts in a way that feels authentic and true to {data['name']}'s character. Allow the conversation to flow naturally, and avoid providing abrupt or disconnected responses.\n\nYou are now {data['name']}. Respond to the therapist's prompts as {data['name']} would, regardless of the specific questions asked. Limit each of your responses to a maximum of 5 sentences. If the therapist begins the conversation with a greeting like \"Hi,\" initiate the conversation as the patient."
                    prompt = f'Review the diagnose from two doctors.\nDiagnose from Doctor 1: {diagnose_1}\nDiagnose from Doctor 2: {diagnose_2}\nExplain which diagnose is more accurate according to your symptoms and why.'
                    if '7b' in model_id:
                        patient_sure, cur_his = getResponse(base_model, sys + '\n' + prompt, [], 'Patient')
                    else:
                        patient_sure, cur_his = getResponse(base_model, prompt, [], 'Patient', sys)
                    print('Patient: ', patient_sure)

                    prompt = f'Refer to the following sentences and answer which diagnose is more accurate. Just answer without explanation:\n{patient_sure}'
                    illness_final, cur_his = getResponse(base_model, prompt, [], 'Patient')
                    print('Patient: ', illness_final)

                    best_treatment, max_score = try_and_reflect(data, illness_final, history, 'treatment')
                    best_medicine, max_score_m = try_and_reflect(data, illness_final, history, 'medicine')

                    best_treatment = best_treatment + '\n' + best_medicine

                    prompt = f'Please extract proper treatment, medicine and corresponding explanation from the following sentences.\n\nOutput Format:\nTreatment: \nExplanation\nMedicine: \nExplanation\n\nSentences: {best_treatment}'
                    output, cur_his = getResponse('gpt-4-turbo-2024-04-09', prompt, [], 'Patient')
                    print(output)
                    output = output.strip()
                    out_list = output.split('\n')
                    treatment = ''
                    explain_t = ''
                    medicine = ''
                    explain_m = ''
                    for i in range(len(out_list)):
                        text = out_list[i].strip()
                        if 'Treatment:' in text:
                            if len(text) > 10:
                                treatment = text[11:].strip()
                            elif i < len(out_list)-1:
                                treatment = out_list[i+1].strip()
                        elif 'Medicine:' in text:
                            if len(text) > 9:
                                medicine = text[10:].strip()
                            elif i < len(out_list)-1:
                                medicine = out_list[i+1].strip()
                        elif 'Explanation:' in text:
                            explain = ''
                            if len(text) > 12:
                                explain = text[13:].strip()
                            elif i < len(out_list)-1:
                                explain = out_list[i+1].strip()

                            if explain_t == '':
                                explain_t = explain
                            else:
                                explain_m = explain

                    # Which of the following is the most likely diagnosis in this patient?
                    # illness_final
                    symptom_f = f"Patient History: {data['history']}\n\nCognitive Conceptualization Diagram:\nIntermediate Beliefs: {data['intermediate_belief']}\nIntermediate Beliefs during Depression: {data['intermediate_belief_depression']}\nCoping Strategies: {data['coping_strategies']}\n\nThe experience in a week.\nSituation: {data['situation']}\nAutomatic Thoughts: {data['auto_thought']}\nEmotions: {data['emotion']}\nBehavior: {data['behavior']}"
                    data_generate = dict()
                    data_generate['symptom'] = symptom_f
                    data_generate['illness'] = illness_final
                    data_generate['treatment'] = treatment
                    data_generate['explain_t'] = explain_t
                    data_generate['medicine'] = medicine
                    data_generate['explain_m'] = explain_m

                    writer.write(data_generate)


if __name__ == '__main__':
    fire.Fire(run)  
                    
