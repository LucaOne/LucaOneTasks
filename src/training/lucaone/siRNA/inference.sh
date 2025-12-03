# input file format(csv, the first row is csv-header), Required columns: seq_id, seq_type, seq, label(可选)
# 如果推理的文件中包含ground truth 列（也就是label列，可以指定--ground_truth_idx 列号(0-based), 在结果文件中会带上ground truth

cd LucaOneTasks/src/
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python predict_v2.py \
    --input_file  ../data/siRNA/binary_class/xxxx.csv  \
    --llm_truncation_seq_length 1024 \
    --model_path .. \
    --save_path ../predicts/siRNA/binary_class/result_xxxx.csv \
    --dataset_name siRNA \
    --dataset_type rna \
    --task_type binary_class \
    --task_level_type seq_level \
    --model_type lucasingle \
    --input_type matrix \
    --input_mode single \
    --time_str 20251120174701 \
    --step 1728432 \
    --print_per_num 1000 \
    --threshold 0.5 \
    --gpu_id 0