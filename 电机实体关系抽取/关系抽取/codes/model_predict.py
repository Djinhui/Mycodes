# coding:utf-8

from config import *
from model import *
import torch
from utils import *

samples = ["即可求得轴系各薄弱截面的疲劳寿命损伤及累计疲劳寿命损伤。"]
conf = Config()
text = conf.tokenizer.batch_encode_plus(samples,
                                        padding='max_length',
                                        max_length=200,
                                        truncation=True)
# 文本处理
input_ids = torch.tensor(text['input_ids']).to(conf.device)
mask = torch.tensor(text['attention_mask']).to(conf.device)

# 初始化主实体长度
sub_len = []
sub_head2tail = []

for i in range(len(samples)):
    sub_len.append(torch.tensor([1],dtype=torch.float))
    sub_head2tail.append(torch.zeros(200))

print('sub_len--->',sub_len)
print('sub_head2tail--->',sub_head2tail)

# 将sub_len和sub_head2tail转换成tensor类型
sub_len = torch.stack(sub_len).to(conf.device)
sub_head2tail = torch.stack(sub_head2tail).to(conf.device)

# 准备数据
inputs = {'input_ids': input_ids,
          'mask': mask,
          'sub_head2tail': sub_head2tail,
          'sub_len': sub_len}

print('准备好的数据--->',inputs)

# 实例化模型
mymodel = CasRel(conf).to(conf.device)
mymodel.load_state_dict(torch.load('../save_model/best_new_f1.pth'))
mymodel.eval()

with torch.no_grad():
    # 从文件中获取关系类型与id的对应关系
    with open(conf.rel_dict_path,'r',encoding='utf-8') as fr:
        relid2word = json.load(fr)
    # 对数据进行模型推理
    logist = mymodel(**inputs)
    print('logist--->',logist)
    # 将预测结果转换成0和1的形式
    pred_sub_heads = convert_score_to_zero_one(logist['pred_sub_heads'])
    pred_sub_tails = convert_score_to_zero_one(logist['pred_sub_tails'])
    batch_size = inputs['input_ids'].shape[0]
    print('batch_size--->',batch_size)
    pred_obj_heads = convert_score_to_zero_one(logist['pred_obj_heads'])
    pred_obj_tails = convert_score_to_zero_one(logist['pred_obj_tails'])

    # 循环遍历每个样本
    for batch_index in range(len(samples)):
        new_dict = {}
        spo_list = []
        ids = inputs['input_ids'][batch_index]
        # 将输入的id序列转换回文本
        text_list = conf.tokenizer.convert_ids_to_tokens(ids)
        # 获取句子的结束位置
        last_index = text_list.index('[SEP]')
        # 提取句子文本
        sentence = ''.join(text_list[1:last_index])
        # 提取出预测得到的主语和宾语的集合
        pred_subs = extract_sub(pred_sub_heads[batch_index].squeeze(),pred_sub_tails[batch_index].squeeze())
        print('主语集合--->',pred_subs)
        pred_objs = extract_obj_and_rel(pred_obj_heads[batch_index],pred_obj_tails[batch_index])
        print('宾语集合--->',pred_objs)

        # 若没提取结果，则跳过
        if len(pred_subs) == 0 or len(pred_objs) == 0:
            print('没有识别出结果')
        # 如果宾语集合比主语集合多，则将主语集合重复至于宾语集合一样多的长度
        if len(pred_objs) > len(pred_subs):
            pred_subs = pred_subs * len(pred_objs)

        # 循环遍历每个主语和对应关系的宾语
        for sub,rel_obj in zip(pred_subs,pred_objs):
            sub_spo = {}

            # 提取出主语
            sub_head,sub_tail = sub
            sub = ''.join(text_list[sub_head: sub_tail + 1])

            # 如果主语中包含'[PAD]',则跳过
            if '[PAD]' in sub:
                continue
            # 将主语加入都spo字典
            sub_spo['subject'] = sub
            # 获取关系类型
            relation = relid2word[str(rel_obj[0])]

            # 提取出宾语
            obj_head,obj_taill = rel_obj[1],rel_obj[2]
            obj = ''.join(text_list[obj_head: obj_taill + 1])
            # 如果宾语中包含'[PAD]',则跳过
            if '[PAD]' in obj:
                continue
            # 将关系类型和宾语加入到spo字典
            sub_spo['predicate'] = relation
            sub_spo['object'] = obj

            # 将此spo字典加入到spo_list
            spo_list.append(sub_spo)
        # text和sop_list加入一个新的字典中，打印出结果
        new_dict['text'] = sentence
        new_dict['spo_list'] = spo_list
        print(new_dict)



