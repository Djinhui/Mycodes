from datasets import load_dataset


# 在线加载


dataset = load_dataset(path='glue', name='sst2', split='train')
print(dataset)  
'''
Dataset({
features: ['sentence', 'label', 'idx'],
num_rows: 67349
})
'''

dataset = load_dataset(path='seamew/ChnSentiCorp') # 云端加载or本地
print(dataset)

'''
DatasetDict({
train: Dataset({
features: ['text', 'label'],
num_rows: 9600
})
validation: Dataset({
features: ['text', 'label'],
num_rows: 0
})
test: Dataset({
features: ['text', 'label'],
num_rows: 1200
})
})
'''


# 保持到本地
dataset.save_to_disk(dataset_dict_path='./datapath/ChnSentiCorp')

# 从本地磁盘加载
from datasets import load_from_disk
dataset = load_from_disk(dataset_dict_path='./datapath/ChnSentiCorp')

# 取数据
dataset_Train = dataset['train']
for i in range(5):
    print(dataset_Train[i])

'''
{'text': '轻便，方便携带，性能也不错，能满足平时的工作需要，对出差人员来讲非常不错',
'label': 1}
{'text': '很好的地理位置，一塌糊涂的服务，萧条的酒店。', 'label': 0}
{'text': '非常不错，服务很好，位于市中心区，交通方便，不过价格也高！', 'label': 1}
{'text': '跟住招待所没什么太大区别。绝对不会再住第2次的酒店！', 'label': 0}
{'text': '价格太高，性价比不够好。我觉得今后还是去其他酒店比较好。', 'label': 0}
'''

# 排序

#数据中的label是无序的
print(dataset_Train['label'][:10])
#让数据按照label排序
sorted_datasetTrain = dataset_Train.sort('label')
print(sorted_datasetTrain['label'][:10])
print(sorted_datasetTrain['label'][-10:])
'''
[1, 1, 0, 0, 1, 0, 0, 0, 1, 1]
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
'''

# 打乱
shuffled_datasetTrain = sorted_datasetTrain.shuffle(seed=42)
print(shuffled_datasetTrain['label'][:10])

# 数据抽样
sampled_datasetTrain = shuffled_datasetTrain.select([0,10,23,45,36])
print(sampled_datasetTrain)
'''
Dataset({
features: ['text', 'label'],
num_rows: 5
})
'''

# 数据过滤
def fun(data):
    return data['text'].startswith('非常')

dataset.filter(fun)


# 训练测试拆分
dataset_Train.train_test_split(test_size=0.2)
'''
DatasetDict({
train: Dataset({
features: ['text', 'label'],
num_rows: 8640
})
test: Dataset({
features: ['text', 'label'],
num_rows: 960
})
})
'''

# 重命名字段
dataset.rename_column('text','sentence')

# 删除字段
dataset.remove_columns(['label'])

# 映射
def fun(data):
    data['text'] = 'My Sentence: ' + data['text']
    return data

maped_dataset = dataset.map(fun)

# 设置数据格式
dataset.set_format(type='torch',columns=['label'],output_all_columns=True)

# 保存其他格式
dataset = load_dataset(path='seamew/ChnSentiCorp', split='train')
dataset.to_csv('./datapath/ChnSentiCorp.csv')
csv_dataset = load_dataset(path='csv', 
                           data_files='./datapath/ChnSentiCorp.csv',
                           split='train')

dataset.to_json('./datapath/ChnSentiCorp.json')
json_dataset = load_dataset(path='json',
                           data_files='./datapath/ChnSentiCorp.json',
                           split='train')
