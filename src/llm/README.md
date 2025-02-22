# LLM for embedding     

### LucaOne(LucaGPLM) Embedding   
for `gene` or `prot`

**Suggestions and Instructions:**
1) Try to use a large GPU-memory machine for embedding reasoning, such as A100, H100, H200, etc., so that long sequences can be processed once.       
   LucaOne can process sequences of about `2800` in length at one time under A100;
2) For long sequences, LucaOne will do overlapped fragments in the sequence for embedding and finally merge them into a completed embedding matrix.        
   Please set `--embedding_complete` and `--embedding_complete_seg_overlap`;
3) If the GPU memory is not enough to process the longer sequence, it will use the CPU for embedding, so the speed will be reduced.       
   If your dataset is small, then you can set: `--gpu_id -1`;
4) If your dataset includes a lot of long sequences (more than 10,000 sequences), please set: `--embedding_complete`, `--embedding_complete_seg_overlap`, and `--embedding_fixed_len_a_time` (represent the maximum length for embedding at one-time).       
   If the sequence length is greater than the value of `--embedding_fixed_len_a_time`, fragment embedding is performed based on this value, and finally, the merge is performed; otherwise, according to the actual length of the sequence;
5) If `--embedding_complete` is not set, the code will truncate the sequence embedding according to the value of `--truncation_seq_length`;
6) For proteins, the length of most proteins is less than 1000; there are not many ultra-long protein sequences, so the value of `--embedding_fixed_len_a_time` can be set a large value or not be set;
7) For DNA, the DNA sequence of many tasks is very long; please set `--embedding_fixed_len_a_time`.  
   The larger the amount of ultra-long sequence, the smaller value should be set, such as `2800` under A100.      
   If the GPU embedding fails to process the longer sequence, the CPU will be called.      
   When the amount of dataset is not large, the spent time will not be long;
8) For RNA, most RNA is not very long, so the processing method can be consistent with the protein, so the `--embedding_fixed_len_a_time` can be set a larger value or not be set.


```
# for DNA or RNA
cd ./src/llm/lucagplm
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucaone_gplm \
    --llm_version v2.0 \
    --llm_task_level token_level,span_level,seq_level,structure_level \
    --llm_time_str 20231125113045 \
    --llm_step 5600000 \
    --truncation_seq_length 100000 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../../../data/test_data/gene/test_gene.fasta \
    --save_path ../../../embedding/lucaone/test_data/gene/test_gene \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 0
    
cd ./src/llm/lucagplm
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8"
python get_embedding.py \
    --llm_dir ../../..  \
    --llm_type lucaone_gplm \
    --llm_version v2.0 \
    --llm_task_level token_level,span_level,seq_level,structure_level \
    --llm_time_str 20231125113045 \
    --llm_step 5600000 \
    --truncation_seq_length 100000 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../data/test_data/prot/test_prot.fasta \
    --save_path ../../../embedding/lucaone/test_data/prot/test_prot \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 0
```

### ESM2 Embedding
only for `prot`

```shell
cd src/llm/esm
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python predict_embedding.py \
    --llm_type esm2 \
    --llm_version 3B \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../data/prot.fasta \
    --save_path ../../../matrices/esm2/prot/ \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 0
```

### DNABert2 Embedding
only for `gene`

```shell
cd src/llm/dnabert2
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python inference_embedding.py \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../../../data/gene.fasta \
    --save_path ../../../../matrices/dnabert2/gene/ \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --gpu_id 0
```

### DNABerts  Embedding
only for `gene`

```shell
cd src/llm/dnaberts
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python inference_embedding.py \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../../../data/gene.fasta \
    --save_path ../../../../matrices/dnaberts/gene/ \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --gpu_id 0
```