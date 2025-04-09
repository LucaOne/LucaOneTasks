# LLM for embedding     

### LucaOne(LucaGPLM) Embedding   
for `gene` or `prot`

**建议与说明:**         
1）尽量使用显存大进行embedding 推理，如：A100，H100，H200等，这样一次性能够处理较长的序列，LucaOne在A100下可以一次性处理`2800`左右长度的序列；   
2）对于超长序列，LucaOne会进行Overlap分片进行embedding，最后合并成完整的embedding，请设置`--embedding_complete`与`--embedding_complete_seg_overlap`；    
3）如果显卡不足以处理输入的序列长度，会调用CPU进行处理，这样速度会变慢，如果你的数据集中长序列不是很多，那么可以使用这种方式: `--gpu_id -1`；      
4）如果你的数据集中长序列很多，比如: 万条以上，那么再设置`--embedding_complete`与`--embedding_complete_seg_overlap`之外，再加上设置`--embedding_fixed_len_a_time`，表示一次性embedding的最大长度。
如果序列长度大于这个长度，基于这个长度进行分片embedding，最后进行合并。否则根据序列的实际长度；    
5）如果不设置`--embedding_complete`，那么根据设置的`--truncation_seq_length`的值对序列进行截断embedding；  
6）对于蛋白，因为绝大部分蛋白长度在1000以下，因此超长蛋白序列不会很多，因此可以将`--embedding_fixed_len_a_time`设置长一点或者`不设置`；    
7）对于DNA，因为很多任务的DNA序列很长，那么请设置`--embedding_fixed_len_a_time`。    
超长序列数据量越多，该值设置越小一点，比如在A100下设置为`2800`，否则设置大一点，如果GPU根据这个长度embedding失败，则会调用CPU。如果数据集数不大，则时间不会很久；          
8）对于RNA，因为大部分RNA不会很长，因此与蛋白处理方式一致，因此可以将`--embedding_fixed_len_a_time`设置长一点或者不设置；

```
# for DNA or RNA
cd ./src/llm/lucagplm

## using lucaone
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8"
python get_embedding.py \
    --llm_dir ../../../  \
    --llm_type lucaone \
    --llm_version lucaone \
    --llm_step 36000000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../../../data/test_data/gene/test_gene.fasta \
    --save_path ../../../embedding/lucaone/test_data/gene/test_gene/36000000/ \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 0
    
    
## using lucaone-gene
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8"
python get_embedding.py \
    --llm_dir ../../../  \
    --llm_type lucaone \
    --llm_version lucaone-gene \
    --llm_step 36800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../../../data/test_data/gene/test_gene.fasta \
    --save_path ../../../embedding/lucaone-gene/test_data/gene/test_gene/36800000/ \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 1
    
    
cd ./src/llm/lucagplm 
## using lucaone
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8"
python get_embedding.py \
    --llm_dir ../../../  \
    --llm_type lucaone \
    --llm_version lucaone \
    --llm_step 36000000 \
    --truncation_seq_length 4096 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../data/test_data/prot/test_prot.fasta \
    --save_path ../../../embedding/lucaone/test_data/prot/test_prot/36000000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 2  
    
    
## using lucaone-prot
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8"
python get_embedding.py \
    --llm_dir ../../../  \
    --llm_type lucaone \
    --llm_version lucaone-prot \
    --llm_step 30000000 \
    --truncation_seq_length 4096 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../../../data/test_data/prot/test_prot.fasta \
    --save_path ../../../embedding/lucaone/test_data/prot/test_prot/30000000 \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 3  
```

### ESM2 Embedding
only for `prot`     

```shell
cd src/llm/esm
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python predict_embedding.py \
    --llm_type esm2 \
    --llm_version 3B \
    --truncation_seq_length 4096 \
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