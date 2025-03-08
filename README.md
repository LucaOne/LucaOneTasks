# Downstream Tasks of LucaOne   

## TimeLine   
* 2024/10/01: optimized embedding inference code: `src/llm/lucagplm/get_embedding.py`       
* 2024/08/01: add `checkpoint=17600000`, location: <a href='http://47.93.21.181/lucaone/TrainedCheckPoint/models/lucagplm/v2.0/token_level,span_level,seq_level,structure_level/lucaone_gplm/20231125113045/checkpoint-step17600000/'>checkpoint-step17600000</a>      
This project will download the checkpoint automatically according to the value of parameter **--llm_step**.   

  
## 1. Networks    
Three distinct networks correspond to three different types of inputs: 
* LucaBase(Single)   
* LucaPPI(Homogeneous Pair)       
* LucaPPI2(Heterogeneous Pair)   


<center>
<img alt="Downstream task network with three input types and results comparison of 8 ver-
ification tasks" src="./pics/DownstreamNetworksAndMetrics.png"/>

Fig. 1 Downstream task network with three input types and results comparison of 8 ver-
ification tasks.   
</center>


* Central Dogma(Central Dogma of Molecular Biology)    
  Input: DNA + Protein(heterogeneous double sequence)          
  Network: LucaPPI2(`src/ppi/models/LucaPPI2`)    


* SupKTax(Genus Taxonomy Annotation)     
  Input: DNA(single sequence)       
  Network: LucaBase(`src/common/luca_base`)   


* GenusTax(SuperKingdom Taxonomy Annotation)     
  Input: DNA(single sequence)       
  Network: LucaBase(`src/common/luca_base`)   


* SpeciesTax(Species Taxonomy Annotation)    
  Input: DNA(single sequence)       
  Network: LucaBase(`src/common/luca_base`)   


* ProtLoc(Prokaryotic Protein Subcellular Location)    
  Input: Protein(single sequence)       
  Network: LucaBase(`src/common/luca_base`)   


* ProtStab(Protein Stability)     
  Input: Protein(single sequence)       
  Network: LucaBase(`src/common/luca_base`)   


* ncRNAFam(Non-coding RNA Family)     
  Input: RNA(single sequence)         
  Network: LucaBase(`src/common/luca_base`)   


* InfA(Influenza A Antigenic Relationship Prediction)   
  Input: RNA + RNA(homogeneous double sequence)         
  Network: LucaPPI(`src/ppi/models/LucaPPI`)    


* PPI(Protein-Protein Interaction)     
  Input: Protein + Protein(homogeneous double sequence)           
  Network: LucaPPI(`src/ppi/models/LucaPPI`)   


* ncRPI(ncRNA-Protein Interactions)          
  Input: DNA + Protein(heterogeneous double sequence)          
  Network: LucaPPI2(`src/ppi/models/LucaPPI2`)

## 2. Environment Installation
### step1: update git
#### 1) centos
sudo yum update     
sudo yum install git-all

#### 2) ubuntu
sudo apt-get update     
sudo apt install git-all

### step2: install python 3.9
#### 1) download anaconda3
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh

#### 2) install conda
sh Anaconda3-2022.05-Linux-x86_64.sh
##### Notice: Select Yes to update ~/.bashrc
source ~/.bashrc

#### 3) create a virtual environment: python=3.9.13
conda create -n lucaone_tasks python=3.9.13

#### 4) activate lucaone_tasks
conda activate lucaone_tasks

### step3:  install other requirements
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple


### for DNABert2 Embedding
**Notice：** Need to switch the virtual environment        

activate deactivate

conda create -n lucaone_tasks_dnabert2 python=3.9.13

conda activate lucaone_tasks_dnabert2

pip install -r requirements_dnabert2.txt -i https://pypi.tuna.tsinghua.edu.cn/simple


## 3. Datasets   
Downstream Tasks Dataset FTP: <a href='http://47.93.21.181/lucaone/DownstreamTasksDataset/dataset/'>Dataset for LucaOneTasks</a>

Copy the 10 datasets from <href> http://47.93.21.181/lucaone/DownstreamTasksDataset/dataset/* </href> into the directory `./dataset/`



## 4. LucaOne Trained Checkpoint    
Trained LucaOne Checkpoint FTP: <a href='http://47.93.21.181/lucaone/TrainedCheckPoint/'>TrainedCheckPoint for LucaOne</a>        

**Notice**    
The project will download automatically LucaOne Trained-CheckPoint from **FTP**.     

When downloading automatically failed, you can manually download:       

Copy the **TrainedCheckPoint Files(`models/` + `logs/`)** from <href> http://47.93.21.181/lucaone/TrainedCheckPoint/* </href> into the directory `./llm/`


## 5. Usage of LucaOne Embedding(can also use LucaOneApp project)      
Methods of using embedding:    
In this project, the sequence is embedded during the training downstream task(`./src/encoder.py`).   

We can also embed the dataset and store into a predefined folder, then build and train the downstream network.   
the script of embedding a dataset(`./src/llm/lucagplm/get_embedding.py`):        

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


### 1) the **csv** file format of input     

**Notice:**       
a. need to specify the column index of the sequence id(*id_idx**) and sequence(**seq_idx**), starting index: 0.        
b. The **sequence id** must be globally unique in the input file and cannot contain special characters (because the embedding file stored is named by the sequence id).     

```shell
# for protein
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
    --input_file ../../../data/test_data/prot/test_prot.csv \
    --id_idx 2 \
    --seq_idx 3 \
    --save_path ../../../embedding/lucaone/test_data/prot/test_prot \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 0   
 
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
    --input_file ../../../data/test_data/gene/test_gene.csv \
    --id_idx 0 \
    --seq_idx 1 \
    --save_path ../../../embedding/lucaone/test_data/gene/test_gene \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id 0   
```

### 2) the **fasta** file format of input       
**Notice:**     
a. The **sequence id** must be globally unique in the input file and cannot contain special characters (because the embedding file stored is named by the sequence id).    

```shell
# for protein
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


```shell
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
```

### 3) Parameters       
1) LucaOne checkpoint parameters:
    * llm_dir: the path for storing the checkpoint LucaOne model，default: `../../../`         
    * llm_type: the type of LucaOne, default: lucagplm         
    * llm_version: the version of LucaOne, default: v2.0         
    * llm_task_level: the pretrained tasks of LucaOne, default: token_level,span_level,seq_level,structure_level          
    * llm_time_str: the trained time str of LucaOne, default: 20231125113045         
    * llm_step:  the trained checkpoint of LucaOne, default: 5600000, choices=[5600000, 17600000]       

2) Important parameters:        
    * embedding_type: `matrix` or `vector`, output the embedding matrix or [CLS] vector for the entire sequence, recommend: matrix.      
    * trunc_type: truncation type: `right` or `left`, truncation when the sequence exceeds the maximum length.    
    * truncation_seq_length: the maximum length for embedding(not including [CLS] and [SEP]), itself does not limit the length, depending on the capacity of GPU.            
    * matrix_add_special_token: if the embedding is matrix, whether the matrix includes [CLS] and [SEP] vectors.           
    * seq_type: type of input sequence: `gene` or `prot`, `gene` for nucleic acid(DNA or RNA), `prot` for protein.        
    * input_file: the input file path for embedding(format: csv or fasta). The seq_id in the file must be unique and cannot contain special characters.     
    * save_path: the saving dir for storing the embedding file.     
    * embedding_complete: When `embedding_complete` is set, `truncation_seq_length` is invalid. If the GPU memory is not enough to infer the entire sequence at once, it is used to determine whether to perform segmented completion (if this parameter is not used, 0.95*len is truncated each time until the CPU can process the length).
    * embedding_complete_seg_overlap: When `embedding_complete` is set, whether the method of overlap is applicable to segmentation(overlap sliding window)
    * embedding_fixed_len_a_time: When the input sequence is too long for your GPU to complete the inference at once, you can specify the fixed length of the inference at once(default: None)    
    * gpu_id: the gpu id to use(-1 for cpu).  

3) Optional parameters:      
    * id_idx & seq_idx: when the input file format is csv file, need to use `id_idx` and `seq_idx` to specify the column index in the csv (starting with 0).   

## 6. Usage of Downstream Models Inference
The <a href='http://47.93.21.181/lucaone/DownstreamTasksTrainedModels/'> FTP</a> includes the trained models of 10 downstream tasks(presented in the paper, all metrics in `TableS5`), all these trained models are based on LucaOne's embedding.        
Use the script `src/predict.py` or `src/predict.sh` to load the trained model and predict.

**Notice**    
The project will download automatically Trained-CheckPoint of all downstream tasks from **FTP**.

When downloading automatically failed, you can manually download:

Copy the **DownstreamTasksTrainedModels Files(`models/` + `logs/`)** from <href> http://47.93.21.181/lucaone/DownstreamTasksTrainedModels/ </href> into the project `LucaOneTasks/`


The shell script of all downstream task models for inference in `LucaOneTasks/src/predict.sh`

```shell 
cd LucaOneTasks/src/
# input file format(csv, the first row is csv-header), Required columns: seq_id_a, seq_id_b, seq_type_a, seq_type_b, seq_a, seq_b
# seq_type_a must be gene, seq_type_a must be prot
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python predict.py \
    --input_file ../test/CentralDogma/CentralDogma_prediction.csv \
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
    --gpu_id 0
```

## 7. Downstream Tasks        
The running scripts of 10 downstream tasks in three directories:      

1) `src/training/lucaone` :     
   The running scripts of the 10 downstream tasks were based on LucaOne's embedding **(Fig. 4 in our paper)**.


2) `src/training/downstream_tasks` :     
   A complete comparison on the 10 downstream tasks.   
   These comparisons were based on the embedding of LucaOne, DNABert2, and ESM2-3B. **(Table.S5 in our paper)**.    


3) `src/training/lucaone_separated` :      
   The task script with the embedding based on the LucaOne of separated nucleic acid and protein training(LucaOne-Gene/LucaOne-Prot). **(Fig. 3 in our paper)**.      


## 8. Data and Code Availability
**FTP:**   
Pre-training data, code, and trained checkpoint of LucaOne, embedding inference code, downstream validation tasks data & code, and other materials are available: <a href='http://47.93.21.181/lucaone/'>FTP</a>.

**Details:**

The LucaOne's model code is available at: <a href='https://github.com/LucaOne/LucaOne'>LucaOne Github </a> or <a href='http://47.93.21.181/lucaone/LucaOne/'>LucaOne</a>.

The trained-checkpoint files are available at: <a href='http://47.93.21.181/lucaone/TrainedCheckPoint/'>TrainedCheckPoint</a>.

LucaOne's representational inference code is available at: <a href='https://github.com/LucaOne/LucaOneApp'>LucaOneApp Github</a> or <a href='http://47.93.21.181/lucaone/LucaOneApp'>LucaOneApp</a>.

The project of 8 downstream tasks is available at: <a href='https://github.com/LucaOne/LucaOneTasks'>LucaOneTasks Github</a> or <a href='http://47.93.21.181/lucaone/LucaOneTasks'>LucaOneTasks</a>.

The pre-training dataset of LucaOne is opened at: <a href='http://47.93.21.181/lucaone/PreTrainingDataset/'>PreTrainingDataset</a>.

The datasets of downstream tasks are available at: <a href='http://47.93.21.181/lucaone/DownstreamTasksDataset/'> DownstreamTasksDataset </a>.

The trained models of downstream tasks are available at: <a href='http://47.93.21.181/lucaone/DownstreamTasksTrainedModels/'> DownstreamTasksTrainedModels </a>.

Other supplementary materials are available at: <a href='http://47.93.21.181/lucaone/Others/'> Others </a>.



## 9. Contributor        
<a href="https://scholar.google.com.hk/citations?user=RDbqGTcAAAAJ&hl=en" title="Yong He">Yong He</a>,
<a href="https://scholar.google.com/citations?user=lT3nelQAAAAJ&hl=en" title="Zhaorong Li">Zhaorong Li</a>,
<a href="https://scholar.google.com/citations?view_op=list_works&hl=en&user=uvrzUfEAAAAJ" title="Yongtao Shan">Yongtao Shan</a>, Yanhong Wei,
<a href="https://scholar.google.com.hk/citations?hl=zh-CN&pli=1&user=Zhlg9QkAAAAJ" title="Yuan-Fei Pan">Yuan-Fei Pan</a>,
<a href="https://scholar.google.com/citations?user=1KJOH7YAAAAJ&hl=zh-CN&oi=ao" title="Mang Shi">Mang Shi</a>


## 10. Zenodo        
We have uploaded the model code, training scripts, and embedding inference scripts of LucaOne;    
The mode code, training and evaluation scripts, datasets, and trained models for downstream tasks,    
and additional supplementary materials to Zenodo (10.5281/zenodo.14977739).    
However, due to the substantial size of the pretraining dataset of LucaOne, it has not been included on Zenodo.     
Instead, it remains accessible via our publicly available FTP server (**<a href='http://47.93.21.181/lucaone/PreTrainingDataset/'>LucaOne Pretraining dataset</a>**).     
We are actively seeking an open FTP platform with sufficient storage capacity to host our pretraining dataset.

**<a href='https://doi.org/10.5281/zenodo.14977739'>LucaOne Zenodo</a>**

## 11. Citation
**<a href='https://www.biorxiv.org/content/10.1101/2024.05.10.592927v1'>LucaOne Biorxiv</a>**


@article {LucaOne,                
author = {Yong He and Pan Fang and Yongtao Shan and Yuanfei Pan and Yanhong Wei and Yichang Chen and Yihao Chen and Yi Liu and Zhenyu Zeng and Zhan Zhou and Feng Zhu and Edward C. Holmes and Jieping Ye and Jun Li and Yuelong Shu and Mang Shi and Zhaorong Li},     
title = {LucaOne: Generalized Biological Foundation Model with Unified Nucleic Acid and Protein Language},      
elocation-id = {2024.05.10.592927},        
year = {2024},         
doi = {10.1101/2024.05.10.592927},        
publisher = {Cold Spring Harbor Laboratory},        
URL = {https://www.biorxiv.org/content/early/2024/05/14/2024.05.10.592927 },        
eprint = {https://www.biorxiv.org/content/early/2024/05/14/2024.05.10.592927.full.pdf },        
journal = {bioRxiv}        
}


## 12. LucaTeam

<center>
<img alt="LucaTeam" src="./pics/LucaTeam.jpg"/>

Fig. 5 LucaTeam at the West Lake in Hangzhou.
</center>   
  