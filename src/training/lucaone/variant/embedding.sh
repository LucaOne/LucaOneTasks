 scp -r ../LucaOneTasks/src sanyuan.hy@8.220.195.47:/mnt/sanyuan.hy/workspace/Quinoa/LucaOneTasks/


cd src/llm/lucgplm
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../../  \
    --llm_type lucaone \
    --llm_version lucaone-gene \
    --llm_step 36800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../../../dataset/QuinoaAltitude/gene/multi_class/all_seqs_01_04.fasta \
    --save_path ../../../../matrices/QuinoaAltitude/lucaone/lucaone-gene/36800000/ \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 0


export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../../  \
    --llm_type lucaone \
    --llm_version lucaone-gene \
    --llm_step 36800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../../../dataset/QuinoaAltitude/gene/multi_class/all_seqs_02_04.fasta \
    --save_path ../../../../matrices/QuinoaAltitude/lucaone/lucaone-gene/36800000/ \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 1


export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../../  \
    --llm_type lucaone \
    --llm_version lucaone-gene \
    --llm_step 36800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../../../dataset/QuinoaAltitude/gene/multi_class/all_seqs_03_04.fasta \
    --save_path ../../../../matrices/QuinoaAltitude/lucaone/lucaone-gene/36800000/ \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 2

export CUDA_VISIBLE_DEVICES="0,1,2,3"
python get_embedding.py \
    --llm_dir ../../../  \
    --llm_type lucaone \
    --llm_version lucaone-gene \
    --llm_step 36800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../../../dataset/QuinoaAltitude/gene/multi_class/all_seqs_04_04.fasta \
    --save_path ../../../../matrices/QuinoaAltitude/lucaone/lucaone-gene/36800000/ \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 3