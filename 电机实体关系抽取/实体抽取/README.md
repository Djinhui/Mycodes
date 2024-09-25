# 基于Pytorch的BERT-IDCNN-CRF中文实体识别实现。
运行环境pytorch_latest_p36



# config.py
- bert_model_dir: bert目录
- vocab_file: bert词表文件
- train_file: 训练集
- dev_file: 测试集
- model_path: 载入已有模型参数文件，指定文件名
- save_model_dir: 模型保存文件路径
- max_length: 最大句子长度
- batch_size: batch大小
- epochs: 训练轮数
- tagset_size: 标签数目
- use_cuda: 是否使用cuda

