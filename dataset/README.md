# ZH       
这里存放每个下游任务的数据集，       
存放的路径为：`dataset/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/`，      
其中：     
`$DATASET_NAME`：是数据集或者任务的名称，       
`$DATASET_TYPE`：是数据集的类型，单序列任务包括：`gene`与`protein`；双序列任务包括`gene_gene`, `protein_protein`, `gene_protein`。     
`$TASK_TYPE`：是任务的类型，包括：`binary_class`, `multi_class`, `multi_label`, `regression`。      
数据集的文件使用csv文件格式。    
该目录下包括的文件：      

* label_name与label index的映射文件：`label.txt`   
有表头，对于分类任务，每个标签名称一行，标签名称所在的行号-1表示其label index     
* 训练集(`train/train.csv`)    
* 验证集(`dev/dev.csv`)    
* 测试集(`test/test.csv`)  

每个csv文件都包括表头，    

对于单序列输入任务，则包括4列：`seq_id,seq_type,seq,label`，分别表示：  
* 序列id    
* 序列类型(取值`gene`或者`prot`)   
* 序列    
* 分类标签索引或者回归值        

对于双序列输入任务，则包括7列：`seq_id_a,seq_id_b,seq_type_a,seq_type_b,seq_a,seq_b,label`，分别表示：   
* 序列a的id   
* 序列b的id   
* 序列a类型(取值`gene`或者`prot`)   
* 序列b类型(取值`gene`或者`prot`)   
* 序列a   
* 序列b   
* 分类标签索引或者回归值     

# EN       
This is where the data sets for each downstream task are stored.     
The task dataset dir path is `dataset/$DATASET_NAME/$DATASET_TYPE/$TASK_TYPE/`,     
where,       
`$DATASET_NAME` is the name of the dataset or task,       
`$DATASET_TYPE` is the type of the dataset,          
Single input sequence tasks include: `gene` and `protein`,    
while the pair input sequences tasks include: `gene_gene`, `protein_protein`, and `gene_protein`.             
`$TASK_TYPE` is the type of the task, including: `binary_class`, `multi_class`, `multi_label`, and `regression`.    

The data files are in `csv` file format, including:      
* the label name and the label index mapping file: label.txt      
The file has a header. Each label name is one line for the classification task, and the line number - 1 indicates its label index.   
* training set (train/train.csv)      
* validation set (dev/dev.csv)   
* test set (test/test.csv)   

Each csv file includes a header.   

For single sequence input tasks, the csv file includes 4 columns: `seq_id,seq_type,seq,label`, 
where,    
* the column `seq_id`: the sequence id (unique)   
* the column `seq_type`: the sequence type (value: `gene` or `prot`)     
* the column `seq_type`: the sequence     
* the column `label`: the classification label index or the regression value       

For the dual sequence input task, it includes 7 columns: `seq_id_a,seq_id_b,seq_type_a,seq_type_b seq_a,seq_b,label`, 
* the column `seq_id_a`: the sequence id a (unique)     
* the column `seq_id_b`: the sequence id b (unique)      
* the column `seq_type_a`: the sequence type a (value: `gene` or `prot`)     
* the column `seq_type_b`: the sequence type b (value: `gene` or `prot`)     
* the column `seq_a`: the sequence a     
* the column `seq_b`: the sequence b     
* the column `label`: the classification label index or the regression value     


# The Datasets of LucaOne Downstream Tasks   
LucaOne Downstream Tasks Dataset FTP: <a href='http://47.93.21.181/lucaone/DownstreamTasksDataset/dataset/'>Dataset for LucaOneTasks</a>   

Copy the 10 datasets from <href> http://47.93.21.181/lucaone/DownstreamTasksDataset/dataset/* </href> into the directory `./dataset/`

