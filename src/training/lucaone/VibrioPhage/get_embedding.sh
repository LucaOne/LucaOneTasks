cd src/llm/lucagplm

export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../../  \
    --llm_type lucaone \
    --llm_version lucaone-prot \
    --llm_step 58700000 \
    --truncation_seq_length 2048 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../dataset/VibrioPhage/v1.0/all_prots_01_04.fasta \
    --save_path ../../../../matrices/VibrioPhage/lucaone/lucaone-prot/58700000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 0


export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../../  \
    --llm_type lucaone \
    --llm_version lucaone-prot \
    --llm_step 58700000 \
    --truncation_seq_length 2048 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../dataset/VibrioPhage/v1.0/all_prots_02_04.fasta \
    --save_path ../../../../matrices/VibrioPhage/lucaone/lucaone-prot/58700000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 1


export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../../  \
    --llm_type lucaone \
    --llm_version lucaone-prot \
    --llm_step 58700000 \
    --truncation_seq_length 2048 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../dataset/VibrioPhage/v1.0/all_prots_03_04.fasta \
    --save_path ../../../../matrices/VibrioPhage/lucaone/lucaone-prot/58700000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 2

export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../../  \
    --llm_type lucaone \
    --llm_version lucaone-prot \
    --llm_step 58700000 \
    --truncation_seq_length 2048 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../dataset/VibrioPhage/v1.0/all_prots_04_04.fasta \
    --save_path ../../../../matrices/VibrioPhage/lucaone/lucaone-prot/58700000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 3