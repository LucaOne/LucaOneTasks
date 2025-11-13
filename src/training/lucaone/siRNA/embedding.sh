scp -r ../LucaOneTasks sanyuan.hy@8.130.41.122:/mnt/sanyuan.hy/workspace/siRNA/


# siRNA
cd src/llm/lucgplm
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../../  \
    --llm_type lucaone \
    --llm_version lucaone \
    --llm_step 36000000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type rna \
    --input_file ../../../dataset/siRNA/rna/regression/all_rna_seqs.fasta \
    --save_path ../../../../matrices/siRNA/lucaone/lucaone/30000000/ \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 0
