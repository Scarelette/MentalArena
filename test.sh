python llama_finetune.py --base_model meta-llama/Meta-Llama-3.1-8B-Instruct  --new_name llama3-diagnose-3 --nepoch 3 --data_files data/finetune/llama/diagnose.jsonl
python llama_finetune.py --base_model models/llama3-diagnose-3  --new_name llama3-diagnose-4 --nepoch 1 --data_files data/finetune/llama/diagnose.jsonl
python llama_finetune.py --base_model models/llama3-diagnose-4  --new_name llama3-diagnose-5 --nepoch 1 --data_files data/finetune/llama/diagnose.jsonl
python llama_finetune.py --base_model models/llama3-diagnose-5  --new_name llama3-diagnose-6 --nepoch 1 --data_files data/finetune/llama/diagnose.jsonl

python llama_finetune.py --base_model meta-llama/Meta-Llama-3.1-8B-Instruct  --new_name llama3-medicine-d-t-3 --nepoch 3 --data_files data/finetune/llama/medicine_d_t.jsonl
python llama_finetune.py --base_model models/llama3-medicine-d-t-3  --new_name llama3-medicine-d-t-4 --nepoch 1 --data_files data/finetune/llama/medicine_d_t.jsonl
python llama_finetune.py --base_model models/llama3-medicine-d-t-4  --new_name llama3-medicine-d-t-5 --nepoch 1 --data_files data/finetune/llama/medicine_d_t.jsonl
python llama_finetune.py --base_model models/llama3-medicine-d-t-5  --new_name llama3-medicine-d-t-6 --nepoch 1 --data_files data/finetune/llama/medicine_d_t.jsonl

python llama_finetune.py --base_model meta-llama/Meta-Llama-3.1-8B-Instruct  --new_name llama3-treatment-d-3 --nepoch 3 --data_files data/finetune/llama/treatment_d.jsonl
python llama_finetune.py --base_model models/llama3-treatment-d-3  --new_name llama3-treatment-d-4 --nepoch 1 --data_files data/finetune/llama/treatment_d.jsonl
python llama_finetune.py --base_model models/llama3-treatment-d-4  --new_name llama3-treatment-d-5 --nepoch 1 --data_files data/finetune/llama/treatment_d.jsonl
python llama_finetune.py --base_model models/llama3-treatment-d-5  --new_name llama3-treatment-d-6 --nepoch 1 --data_files data/finetune/llama/treatment_d.jsonl



