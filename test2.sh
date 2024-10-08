for name in "Ophthalmology" "Microbiology" "Pediatrics" "Anatomy" "Medicine" "Pathology" "Skin" "Psychiatry" "ENT" "Pharmacology"
do
    nohup python MedQA_eval.py --model gpt-3.5-turbo-0125 --dataset MedMCQA --name $name &
    nohup python MedQA_eval.py --model ft:gpt-3.5-turbo-0125:robustlearn::9z1AN8GT --dataset MedMCQA --name $name &
done

sleep 200

for name in "Ophthalmology" "Microbiology" "Pediatrics" "Anatomy" "Medicine" "Pathology" "Skin" "Psychiatry" "ENT" "Pharmacology"
do
    nohup python MedQA_eval.py --model gpt-3.5-turbo-0125 --dataset MedMCQA --name $name &
    nohup python MedQA_eval.py --model ft:gpt-3.5-turbo-0125:robustlearn::9z1AN8GT --dataset MedMCQA --name $name &
done

sleep 200

for name in "Ophthalmology" "Microbiology" "Pediatrics" "Anatomy" "Medicine" "Pathology" "Skin" "Psychiatry" "ENT" "Pharmacology"
do
    nohup python MedQA_eval.py --model gpt-3.5-turbo-0125 --dataset MedMCQA --name $name &
    nohup python MedQA_eval.py --model ft:gpt-3.5-turbo-0125:robustlearn::9z1AN8GT --dataset MedMCQA --name $name &
done

sleep 200

for name in "Ophthalmology" "Microbiology" "Pediatrics" "Anatomy" "Medicine" "Pathology" "Skin" "Psychiatry" "ENT" "Pharmacology"
do
    nohup python MedQA_eval.py --model gpt-3.5-turbo-0125 --dataset MedMCQA --name $name &
    nohup python MedQA_eval.py --model ft:gpt-3.5-turbo-0125:robustlearn::9z1AN8GT --dataset MedMCQA --name $name &
done

sleep 200

for name in "Ophthalmology" "Microbiology" "Pediatrics" "Anatomy" "Medicine" "Pathology" "Skin" "Psychiatry" "ENT" "Pharmacology"
do
    nohup python MedQA_eval.py --model gpt-3.5-turbo-0125 --dataset MedMCQA --name $name &
    nohup python MedQA_eval.py --model ft:gpt-3.5-turbo-0125:robustlearn::9z1AN8GT --dataset MedMCQA --name $name &
done
