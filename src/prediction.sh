# CentralDogma
cd LucaOneTasks/src/
# input file format(csv, the first row is csv-header), Required columns: seq_id_a, seq_id_b, seq_type_a, seq_type_b, seq_a, seq_b
# seq_type_a must be gene, seq_type_a must be prot
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python predict_v1.py \
    --input_file  ../test/CentralDogma/CentralDogma_prediction.csv \
    --llm_truncation_seq_length 4096 \
    --model_path .. \
    --save_path ../predicts/CentralDogma/CentralDogma_prediction_results.csv \
    --dataset_name CentralDogma \
    --dataset_type gene_protein \
    --task_type binary_class \
    --task_level_type seq_level \
    --model_type lucappi2 \
    --input_type matrix \
    --input_mode pair \
    --time_str 20240406173806 \
    --print_per_num 1000 \
    --step 64000 \
    --threshold 0.5 \
    --gpu_id 3

# GenusTax
cd LucaOneTasks/src/
# input file format(csv, the first row is csv-header), Required columns: seq_id, seq_type, seq
# seq_type must be gene
# input file format also can be fasta
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python predict_v1.py \
    --input_file ../test/GenusTax/GenusTax_prediction.csv \
    --llm_truncation_seq_length 4096 \
    --model_path .. \
    --save_path ../predicts/GenusTax/GenusTax_prediction_results.csv \
    --dataset_name GenusTax \
    --dataset_type gene \
    --task_type multi_class \
    --task_level_type seq_level \
    --model_type luca_base \
    --input_type matrix \
    --input_mode single \
    --time_str 20240412100337 \
    --print_per_num 1000 \
    --step 24500 \
    --gpu_id 3


# InfA
cd LucaOneTasks/src/
# input file format(csv, the first row is csv-header),  Required columns: seq_id_a, seq_id_b, seq_type_a, seq_type_b, seq_a, seq_b
# seq_type_a must be gene, seq_type_a must be gene
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python predict_v1.py \
    --input_file ../test/InfA/InfA_prediction.csv \
    --llm_truncation_seq_length 4096 \
    --model_path .. \
    --save_path ../predicts/InfA/InfA_prediction_results.csv \
    --dataset_name InfA \
    --dataset_type gene_gene \
    --task_type binary_class \
    --task_level_type seq_level \
    --model_type lucappi \
    --input_type matrix \
    --input_mode pair \
    --time_str 20240214105653 \
    --print_per_num 1000 \
    --step 9603 \
    --threshold 0.5 \
    --gpu_id 3


# ncRNAFam
cd LucaOneTasks/src/
# input file format(csv, the first row is csv-header), Required columns: seq_id, seq_type, seq
# seq_type must be gene
# input file format also can be fasta
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python predict_v1.py \
    --input_file ../test/ncRNAFam/ncRNAFam_prediction.csv \
    --llm_truncation_seq_length 4096 \
    --model_path .. \
    --save_path ../predicts/ncRNAFam/ncRNAFam_prediction_results.csv \
    --dataset_name ncRNAFam \
    --dataset_type gene \
    --task_type multi_class \
    --task_level_type seq_level \
    --model_type luca_base \
    --input_type matrix \
    --input_mode single \
    --time_str 20240414155526 \
    --print_per_num 1000 \
    --step 1958484 \
    --gpu_id 3


# ncRPI
cd LucaOneTasks/src/
# input file format(csv, the first row is csv-header), Required columns: seq_id_a, seq_id_b, seq_type_a, seq_type_b, seq_a, seq_b
# seq_type_a must be gene, seq_type_a must be prot
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python predict_v1.py \
    --input_file ../test/ncRPI/ncRPI_prediction.csv \
    --llm_truncation_seq_length 4096 \
    --model_path .. \
    --save_path ../predicts/ncRPI/ncRPI_prediction_results.csv \
    --dataset_name ncRPI \
    --dataset_type gene_protein \
    --task_type binary_class \
    --task_level_type seq_level \
    --model_type lucappi2 \
    --input_type matrix \
    --input_mode pair \
    --time_str 20240404105148 \
    --print_per_num 1000 \
    --step 716380 \
    --threshold 0.5 \
    --gpu_id 3


# PPI
cd LucaOneTasks/src/
# input file format(csv, the first row is csv-header), Required columns: seq_id_a, seq_id_b, seq_type_a, seq_type_b, seq_a, seq_b
# seq_type_a must be prot, seq_type_a must be prot
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python predict_v1.py \
    --input_file ../test/PPI/PPI_prediction.csv \
    --llm_truncation_seq_length 4096 \
    --model_path .. \
    --save_path ../predicts/PPI/PPI_prediction_results.csv \
    --dataset_name PPI \
    --dataset_type protein \
    --task_type binary_class \
    --task_level_type seq_level \
    --model_type lucappi \
    --input_type matrix \
    --input_mode pair \
    --time_str 20240216205421 \
    --print_per_num 1000 \
    --step 52304 \
    --threshold 0.5 \
    --gpu_id 3

# ProtLoc
cd LucaOneTasks/src/
# input file format(csv, the first row is csv-header), Required columns: seq_id, seq_type, seq
# seq_type must be prot
# input file format also can be fasta
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python predict_v1.py \
    --input_file ../test/ProtLoc/ProtLoc_prediction.csv \
    --llm_truncation_seq_length 4096 \
    --model_path .. \
    --save_path ../predicts/ProtLoc/ProtLoc_prediction_results.csv \
    --dataset_name ProtLoc \
    --dataset_type protein \
    --task_type multi_class \
    --task_level_type seq_level \
    --model_type luca_base \
    --input_type matrix \
    --input_mode single \
    --time_str 20240412140824 \
    --print_per_num 1000 \
    --step 466005 \
    --gpu_id 3


# ProtStab
cd LucaOneTasks/src/
# input file format(csv, the first row is csv-header), Required columns: seq_id, seq_type, seq
# seq_type must be prot
# input file format also can be fasta
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python predict_v1.py \
    --input_file ../test/ProtStab/ProtStab_prediction.csv \
    --llm_truncation_seq_length 4096 \
    --model_path .. \
    --save_path ../predicts/ProtStab/ProtStab_prediction_results.csv \
    --dataset_name ProtStab \
    --dataset_type protein \
    --task_type regression \
    --task_level_type seq_level \
    --model_type luca_base \
    --input_type matrix \
    --input_mode single \
    --time_str 20240404104215 \
    --print_per_num 1000 \
    --step 70371 \
    --gpu_id 3


# SpeciesTax
# input file format(csv, the first row is csv-header), Required columns: seq_id, seq_type, seq
# seq_type must be gene
# input file format also can be fasta
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python predict_v1.py \
    --input_file ../test/SpeciesTax/SpeciesTax_prediction.csv \
    --llm_truncation_seq_length 4096 \
    --model_path .. \
    --save_path ../predicts/SpeciesTax/SpeciesTax_prediction_results.csv \
    --dataset_name SpeciesTax \
    --dataset_type gene \
    --task_type multi_class \
    --task_level_type seq_level \
    --model_type luca_base \
    --input_type matrix \
    --input_mode single \
    --time_str 20240411144916 \
    --print_per_num 1000 \
    --step 24000 \
    --gpu_id 3


# SupKTax
cd LucaOneTasks/src/
# input file format(csv, the first row is csv-header), Required columns: seq_id, seq_type, seq
# seq_type must be gene
# input file format also can be fasta
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python predict_v1.py \
    --input_file ../test/SupKTax/SupKTax_prediction.csv \
    --llm_truncation_seq_length 4096 \
    --model_path .. \
    --save_path ../predicts/SupKTax/SupKTax_prediction_results.csv \
    --dataset_name SupKTax \
    --dataset_type gene \
    --task_type multi_class \
    --task_level_type seq_level \
    --model_type luca_base \
    --input_type matrix \
    --input_mode single \
    --time_str 20240212202328 \
    --print_per_num 1000 \
    --step 37000 \
    --gpu_id 3