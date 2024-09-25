# -*- coding:utf-8 -*-
from config import *


class InputFeatures(object):
    def __init__(self, text, label, input_id, label_id, input_mask):
        self.text = text
        self.label = label
        self.input_id = input_id
        self.label_id = label_id
        self.input_mask = input_mask


# 读取字典文件, 并构造字典
def load_vocab(vocab_file):
    vocab = {}
    index = 0
    with open(vocab_file, 'r', encoding='utf-8',) as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    # print(vocab)
    return vocab


# 加载训练集, 测试集的原始数据
def load_file(file_path):
    contents = open(file_path, encoding='utf-8').readlines()
    text =[]
    label = []
    texts = []
    labels = []
    for inx,line in enumerate(contents):
        if line != '\n':
            line = line.strip().split(' ')
            # 检查是否有漏标的字符
            if len(line) == 1:
                print(f'没有标签',line,inx)
                continue
            text.append(line[0])
            label.append(line[-1])

        else:
            texts.append(text)
            labels.append(label)
            text = []
            label = []
    # print('text--->',text)
    # print('label--->',label)
    return texts, labels


def load_data(file_path, max_length, label_dic, vocab):
    # 载入文本和标签
    texts, labels = load_file(file_path)
    # 确保文本和标签的数量相同
    assert len(texts) == len(labels)
    result = []
    # 遍历所有的文本和标签
    for i in range(len(texts)):

        # 确保文本和标签长度相同
        assert len(texts[i]) == len(labels[i])
        # 获取文本和标签
        token = texts[i]
        label = labels[i]
        # 如果文本超过长度超过最大长度，截取到最大长度-2
        if len(token) > max_length - 2:
            token = token[0: (max_length - 2)]
            label = label[0: (max_length - 2)]
        # if len(texts) != 2:
        #     raise ValueError("Missing label in input file.")

        # 调用BERT的时候需要添加[CLS]和[SEP], 此处仿照同样的规则进行
        tokens_f =['[CLS]'] + token + ['[SEP]']

        # 调用IDCNN的时候添加首尾字符, 以对应上面的CLS和SEP
        label_f = ['<start>'] + label + ['<eos>']

        input_ids = [int(vocab[i]) if i in vocab else int(vocab['[UNK]']) for i in tokens_f]
        # print(label_dic)
        label_ids = [label_dic.get(i, 17) for i in label_f]
        # print(label_f)
        label_ids = [label_dic[i] for i in label_f]

        # 调用BERT的时候需要mask, 调用IDCNN的时候不需要mask
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_length:
            input_ids.append(0)
            input_mask.append(0)
            label_ids.append(label_dic['<pad>'])

        assert len(input_ids) == max_length
        assert len(input_mask) == max_length
        assert len(label_ids) == max_length

        feature = InputFeatures(text=tokens_f, label=label_f, input_id=input_ids,
                                input_mask=input_mask, label_id=label_ids)

        result.append(feature)

    return result


# 利用标签字典恢复真实的预测标签
def recover_label(pred_var, gold_var, l2i_dic, i2l_dic):
    assert len(pred_var) == len(gold_var)  # 检查预测序列和真实序列的长度是否相等
    pred_variable = []
    gold_variable = []
    # 对于每个序列
    for i in range(len(gold_var)):
        # 使用标签到索引字典找到序列的起始和结束索引
        start_index = gold_var[i].index(l2i_dic['<start>'])
        end_index = gold_var[i].index(l2i_dic['<eos>'])
        # 将预测和真实序列在起始和结束索引之间的切片附加到相应的列表中
        pred_variable.append(pred_var[i][start_index:end_index])
        gold_variable.append(gold_var[i][start_index:end_index])
    # 初始化空列表以存储预测和真实标签
    pred_label = []
    gold_label = []
    # 循环每个变量
    for j in range(len(gold_variable)):
        # 使用索引到标签字典将变量从索引转换为标签，并将其附加到预测和真实标签列表中
        pred_label.append([ i2l_dic[t] for t in pred_variable[j] ])
        gold_label.append([ i2l_dic[t] for t in gold_variable[j] ])
    # 返回预测和真实标签列表
    return pred_label, gold_label


# 计算NER的关键指标
def get_ner_fmeasure(golden_lists, predict_lists, label_type='BMES'):
    sent_num = len(golden_lists)
    golden_full = []
    predict_full = []
    right_full = []
    right_tag = 0
    all_tag = 0

    for idx in range(0, sent_num):
        golden_list = golden_lists[idx]
        predict_list = predict_lists[idx]
        
        for idy in range(len(golden_list)):
            if golden_list[idy] == predict_list[idy]:
                right_tag += 1
        all_tag += len(golden_list)
        
        if label_type == 'BMES':
            gold_matrix = get_ner_BMES(golden_list)
            pred_matrix = get_ner_BMES(predict_list)
        else:
            gold_matrix = get_ner_BIO(golden_list)
            pred_matrix = get_ner_BIO(predict_list)
        
        right_ner = list(set(gold_matrix).intersection(set(pred_matrix)))
        golden_full += gold_matrix
        predict_full += pred_matrix
        right_full += right_ner
    
    right_num = len(right_full)
    golden_num = len(golden_full)
    predict_num = len(predict_full)
    
    if predict_num == 0:
        precision = -1
    else:
        precision =  (right_num + 0.0) / predict_num
    
    if golden_num == 0:
        recall = -1
    else:
        recall = (right_num + 0.0) / golden_num
    
    if (precision == -1) or (recall == -1) or (precision + recall) <= 0.:
        f_measure = -1
    else:
        f_measure = 2 * precision * recall / (precision + recall)
    
    accuracy = (right_tag + 0.0) / all_tag
    print ('gold_num = ', golden_num, ' pred_num = ', predict_num, ' right_num = ', right_num)

    return accuracy, precision, recall, f_measure

# 用于将输入的字符串按照规则进行反转并返回结果
def reverse_style(input_string):
    # 获取左括号在字符串中的位置记录在变量target_position中
    target_position = input_string.index('[')
    # 获取字符串的总长度，记录在input_len中
    input_len = len(input_string)
    # 根据指定规则生成反转后的字符串，并储存在output_string中
    output_string = input_string[target_position: input_len] + input_string[0: target_position]
    # 返回反转后的字符串output_string
    return output_string


def get_ner_BMES(label_list):
    list_len = len(label_list)
    begin_label = 'B-'
    end_label = 'E-'
    single_label = 'S'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    
    for i in range(0, list_len):
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i - 1))
            whole_tag = current_label.replace(begin_label, '', 1) + '[' + str(i)
            index_tag = current_label.replace(begin_label, '', 1)

        elif single_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i - 1))
            whole_tag = current_label.replace(single_label, '', 1) + '[' + str(i)
            tag_list.append(whole_tag)
            whole_tag = ''
            index_tag = ''
        elif end_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i))
            whole_tag = ''
            index_tag = ''
        else:
            continue
    
    if (whole_tag != '') & (index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i] + ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    
    return stand_matrix




if __name__ == '__main__':
    pass

