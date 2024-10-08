for model in 'meta-llama/Meta-Llama-3.1-8B-Instruct' 
do
    python MedQA_eval.py --model $model --dataset MedMCQA
    python MedQA_eval.py --model $model --dataset MedQA
    python MedQA_eval.py --model $model --dataset PubMedQA
done

for model in 'gpt-3.5-turbo-0125' 
do
    nohup python MedQA_eval.py --model $model --dataset MedMCQA &
    nohup python MedQA_eval.py --model $model --dataset MedQA &
    nohup python MedQA_eval.py --model $model --dataset PubMedQA &
done

