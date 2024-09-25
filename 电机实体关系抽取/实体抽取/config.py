# -*- coding:utf-8 -*-
# 设定label_to_id的映射字典, 采用BIEO标注模式, 外加3个特殊字符<pad>, <start>, <eos>
l2i_dic = {'O': 0,'S': 1,u'B-sym': 2,u'B-dis': 3,u'B-num': 4,u'B-sce': 5,u'B-uis': 6,u'I-sym': 7,u'I-dis': 8,u'I-num': 9,u'I-sce': 10,u'I-uis': 11,u'E-sym': 12,u'E-dis': 13,u'E-num': 14,u'E-sce': 15,u'E-uis': 16,'<pad>': 17,'<start>': 18,'<eos>': 19}

# 上面的逆向字典, id_to_label
i2l_dic = {0 :'O',
           1 :'S',
           2 :u'B-sym',
           3 :u'B-dis',
           4 :u'B-num',
           5 :u'B-sce',
           6 :u'B-uis',
           7 :u'I-sym',
           8 :u'I-dis',
           9 :u'I-num',
           10 :u'I-sce',
           11 :u'I-uis',
           12 :u'E-sym',
           13 :u'E-dis',
           14 :u'E-num',
           15 :u'E-sce',
           16 :u'E-uis',
           17 :'<pad>',
           18 :'<start>',
           19 :'<eos>'}

# 训练集, 测试集, 词表
train_file = './data/train.txt'
dev_file = './data/test.txt'
vocab_file = './data/vocab.txt'

save_model_path = './saved_model/idcnn_crf.pt'

model_path = './saved_model/idcnn_crf_1.pt'

# 设置关键的超参数
max_length = 256
batch_size = 32
epochs = 150
tagset_size = len(l2i_dic)
dropout = 0.4
use_cuda = True

