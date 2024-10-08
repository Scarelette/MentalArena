import jsonlines
import time
from openai import OpenAI

def revert(input_file, finetune_file):
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


                illness_item = {"messages": [{"role": "system", "content": f"You are a psychiatric expert."}, 
                                                {"role": "user", "content": illness_input}, 
                                                {"role": "assistant", "content": illness_output}]}

                treat_item = {"messages": [{"role": "system", "content": f"You are a psychiatric expert."}, 
                                                {"role": "user", "content": treatment_input}, 
                                                {"role": "assistant", "content": treatment_output}]}
                writer.write(illness_item)
                writer.write(treat_item)
                
                if len(medicine_input) > 0:
                    medicine_item = {"messages": [{"role": "system", "content": f"You are a psychiatric expert."}, 
                                                {"role": "user", "content": medicine_input}, 
                                                {"role": "assistant", "content": medicine_output}]}
                    writer.write(medicine_item)

            print('ok!')    

def fine_tuning_func(api_key, data_file, base_model, n_epochs):
    client = OpenAI(api_key=api_key)
    client.fine_tuning.jobs.create(
        training_file=data_file, 
        model=base_model, 
        hyperparameters={
            "n_epochs": n_epochs
        }
    )


def run(api_key, input_file, finetune_file, base_model, n_epochs):

    revert(input_file, finetune_file)

    fine_tuning_func(api_key, finetune_file, base_model, n_epochs)



if __name__ == '__main__':
    fire.Fire(run)  

