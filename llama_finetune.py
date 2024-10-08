import os
import torch
import fire
from datasets import load_dataset
import jsonlines
from transformers import (
    AutoModelForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer
from accelerate import init_empty_weights,infer_auto_device_map,load_checkpoint_in_model,dispatch_model


def run(base_model, new_name, nepoch, data_files):
    new_model = f'models/{new_name}'
    dataset = load_dataset('json', data_files=data_files, split='train')
    print(dataset)
    print(dataset[0])

    compute_dtype = getattr(torch, "float16")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    max_memory = {i: '46000MB' for i in range(torch.cuda.device_count())}
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        # device_map={"": 0}
        device_map="auto",
        max_memory=max_memory
    )
    model.quantization_config = quant_config
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    peft_params = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )
    training_params = TrainingArguments(
        output_dir="models/results-finetune",
        num_train_epochs=nepoch,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=25,
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="tensorboard"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_params,
        dataset_text_field="text",
        max_seq_length=None,
        tokenizer=tokenizer,
        args=training_params,
        packing=False,
    )
    trainer.train()

    trainer.model.save_pretrained(new_model)
    trainer.tokenizer.save_pretrained(new_model)


def data_process(input_file, finetune_file):
    with jsonlines.open(finetune_file, mode='a') as writer:
        with open(input_file, "r+", encoding="utf8") as f:
            for item in jsonlines.Reader(f):
                symptom = item['symptom']
                illness = item['illness']
                treatment = item['treatment']
                explain_t = item['explain_t']
                medicine = item['medicine']
                explain_m = item['explain_m']

                illness_input = f'{symptom}\nWhat is the most likely diagnosis in this patient?'
                illness_output = illness
                treatment_input = f'{symptom}What is the most appropriate treatment for the patient?'
                treatment_output = f'{treatment}. {explain_t}'
                medicine_input = ''
                word_list = ['None', 'N/A', 'Not', 'Explanation']
                if len(medicine) > 0:
                    tag = 0
                    for word in word_list:
                        if word in medicine:
                            tag = 1
                    if tag == 0:
                        medicine_input = f'{symptom}\nWhat is the proper medication for this patient?'
                        medicine_output = f'{medicine}. {explain_m}'


                illness_item = {"text": f"### System: You are a psychiatric expert. Question: {illness_input}\n ### Answer: {illness_output}"}

                treat_item = {"text": f"### System: You are a psychiatric expert. Question: {treatment_input}\n ### Answer: {treatment_output}"}

                writer.write(illness_item)
                writer.write(treat_item)
                
                if len(medicine_input) > 0:
                    medicine_item = {"text": f"### System: You are a psychiatric expert. Question: {medicine_input}\n ### Answer: {medicine_output}"}

                    writer.write(medicine_item)

            print('ok!')    


def remove_redun(finetune_file, final_finetune_file):
    new_dict = dict()
    with jsonlines.open(final_finetune_file,mode='a') as writer:
        with open(finetune_file, "r+", encoding="utf8") as f:
            for item in jsonlines.Reader(f):
                content = item['text']
                if content not in new_dict.keys():
                    new_dict[content] = 1

            for key in new_dict.keys():
                cur_dict = {'text': key}
                writer.write(cur_dict)

    print('ok!')


# data_process()
# remove_redun()
# train()
# eval()

if __name__ == '__main__':
    fire.Fire(run)  

