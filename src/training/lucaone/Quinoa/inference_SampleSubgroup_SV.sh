cd ../../../../src/



# desc:
# dataset_name: 数据集名称，根据实际情况设置
# dataset_type：数据集数据类型，写死
# task_type: 任务类型，都是多分类，写死
# task_level_type: 任务类型，这里是序列级别分类，写死
# model_path: 模型checkpoint所在一级目录，在src上一层，写死
# model_type: 模型类型，写死
# input_mode: 模型输入模式，single or pair，我们这是single，写死
# input_type: 模型输入类型，LucaOne embedidng matrix + variant变异序列， 如果是基于原始的序列建模，则是seq_variant
# time_str: 模型构建的时间戳，用于拼接得到模型checkpoint的文件路径，写死
# step: 模型checkpoint, 一般使用模型训练日志logs.txt 最后update_flag为true的step，写死
# input_file: 输入文件路径，对应train/val/dev
# llm_truncation_seq_length: 允许的最长序列，我们构造的数据集都是3072+2,所以4096没问题，写死
# output_attention_pooling_scores_dirpath: attention weights保存路径，每一个样本存储一个文件，用torch.load加载
# emb_dir: 推理过程中LucaOne embedding保存路径
# save_path: 推理结果保存路径
# ground_truth_idx: ground truth label 在csv的列号(0-based)，可以没有
# ground_truth_label_idx_2_name: 如果输入文件中有ground_truth，并且ground_truth是label idx，那么可以指定将其转换为label name
# print_per_num: 推理完多少条则print进度信息，并批量写入结果文件
# gpu_id: 调用机器的哪张卡的卡号去推理


# Set these values according to your need
dataset_name="SampleSubgroup_SV"
dataset_type="gene"
task_type="multi_class"
# train, val, test
dataset_split="train"
filename="train.csv"
time_str="20251022202703"
step="128312"
gpu_id=1
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python predict_v2.py \
    --input_file ../dataset/${dataset_name}/${dataset_type}/${task_type}/${dataset_split}/${filename} \
    --llm_truncation_seq_length 4096 \
    --model_path .. \
    --save_path ../predicts/${dataset_name}/${dataset_type}/${task_type}/${dataset_split}/results_${filename} \
    --output_attention_pooling_scores_dirpath ../predicts/${dataset_name}/${dataset_type}/${task_type}/${dataset_split}/attention_pooling_scores/ \
    --emb_dir ../predicts/${dataset_name}/${dataset_type}/${task_type}/${dataset_split}/seq_embedding/ \
    --dataset_name ${dataset_name} \
    --dataset_type ${dataset_type} \
    --task_type ${task_type} \
    --task_level_type seq_level \
    --model_type lucasingle \
    --input_type matrix_variant \
    --input_mode single \
    --time_str ${time_str} \
    --step ${step} \
    --ground_truth_idx 4 \
    --ground_truth_label_idx_2_name \
    3 --print_per_num 10000 \
    --gpu_id ${gpu_id}


