# 抽取式阅读理解（Span-extraction Reading Comprehension）SRC
# 问题Q=q1q2...qn, 篇章P=p1p2...pm
# X =[CLS]q1q2...qn[SEP]p1p2...pm[SEP]

"""
抽取式阅读理解主要由篇章Passage、问题Question和答案Answer构成,要求机器在阅读篇章和问题后给出相应的答案，
而答案要求是从篇章中抽取出的一个文本片段Span。该任务可以简化为预测篇章中的一个起始位置和终止位置而答案就是介于两者之间的文本片段

将问题放在篇章的前面。其原因是BERT一次只能处理一个固定长度为N 的文本序列(如N=512)。
如果将问题放在输入的后半部分,当篇章和问题的总长度超过N 时，部分问题文本将会被截断，导致无法获得完整的问题信息，
进而影响阅读理解系统的整体效果。而将篇章放在后半部分，虽然部分甚至全部篇章文本可能会被截断，
但可以通过篇章切片的方式进行多次预测，并综合相应的答题结果得到最终的输出
"""


import numpy as np
from datasets import load_dataset, load_metric
from evaluate import load
from transformers import BertTokenizerFast, BertForQuestionAnswering, TrainingArguments, Trainer, default_data_collator

# 加载训练数据、分词器、预训练模型以及评价方法
dataset = load_dataset('squad')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
model = BertForQuestionAnswering.from_pretrained('bert-base-cased', return_dict=True)
metric = load_metric('squad')
metric = load('squad')

"""
max_length = 384 
doc_stride = 128 
example = dataset["train"][173] 
tokenized_example = tokenizer(
    example["question"], 
    example["context"], 
    max_length=max_length, 
    truncation="only_second", 
    return_overflowing_tokens=True, 
    stride=doc_stride 
) 

len(tokenized_example['input_ids'])
for input_ids in tokenized_example["input_ids"][:2]: 
    print(tokenizer.decode(input_ids)) 
    print("-"*50)


question + context的长度过长,滑动切分
# 格式
# [CLS]question[SEP]context[:max_length]
# [CLS]question[SEP]context[stride:stride+max_length]
[CLS] beyonce got married in 2008 to whom? [SEP] on april 4, 2008, beyonce married jay z. she publicly revealed their marriage in a video montage at the listening party for her third studio album, i am... sasha fierce, in manhattan's sony club on october 22, 2008. i am... sasha fierce was released on november 18, 2008 in the united states. the album formally introduces beyonce's alter ego sasha fierce, conceived during the making of her 2003 single " crazy in love ", selling 482, 000 copies in its first week, debuting atop the billboard 200, and giving beyonce her third consecutive number - one album in the us. the album featured the number - one song " single ladies ( put a ring on it ) " and the top - five songs " if i were a boy " and " halo ". achieving the accomplishment of becoming her longest - running hot 100 single in her career, " halo "'s success in the us helped beyonce attain more top - ten singles on the list than any other woman during the 2000s. it also included the successful " sweet dreams ", and singles " diva ", " ego ", " broken - hearted girl " and " video phone ". the music video for " single ladies " has been parodied and imitated around the world, spawning the " first major dance craze " of the internet age according to the toronto star. the video has won several awards, including best video at the 2009 mtv europe music awards, the 2009 scottish mobo awards, and the 2009 bet awards. at the 2009 mtv video music awards, the video was nominated for nine awards, ultimately winning three including video of the year. its failure to win the best female video category, which went to american country pop singer taylor swift's " you belong with me ", led to kanye west interrupting the ceremony and beyonce [SEP]
--------------------------------------------------
[CLS] beyonce got married in 2008 to whom? [SEP] single ladies " has been parodied and imitated around the world, spawning the " first major dance craze " of the internet age according to the toronto star. the video has won several awards, including best video at the 2009 mtv europe music awards, the 2009 scottish mobo awards, and the 2009 bet awards. at the 2009 mtv video music awards, the video was nominated for nine awards, ultimately winning three including video of the year. its failure to win the best female video category, which went to american country pop singer taylor swift's " you belong with me ", led to kanye west interrupting the ceremony and beyonce improvising a re - presentation of swift's award during her own acceptance speech. in march 2009, beyonce embarked on the i am... world tour, her second headlining worldwide concert tour, consisting of 108 shows, grossing $ 119. 5 million. [SEP]
--------------------------------------------------
"""

def prepare_train_features(examples, pad_on_right=True): 
    tokenized_examples = tokenizer( 
        examples["question" if pad_on_right else "context"], 
        examples["context" if pad_on_right else "question"], 
        truncation="only_second" if pad_on_right else "only_first", 
        max_length=384, 
        stride=128, 
        return_overflowing_tokens=True, 
        return_offsets_mapping=True, 
        padding="max_length", 
    ) 
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping") 
    offset_mapping = tokenized_examples.pop("offset_mapping") 
    tokenized_examples["start_positions"] = [] 
    tokenized_examples["end_positions"] = [] 
    for i, offsets in enumerate(offset_mapping): 
        input_ids = tokenized_examples["input_ids"][i] 
        cls_index = input_ids.index(tokenizer.cls_token_id) 
        sequence_ids = tokenized_examples.sequence_ids(i) 
        sample_index = sample_mapping[i] 
        answers = examples["answers"][sample_index] 
        if len(answers["answer_start"]) == 0: 
            tokenized_examples["start_positions"].append(cls_index) 
            tokenized_examples["end_positions"].append(cls_index) 
        else: 
            start_char = answers["answer_start"][0] 
            end_char = start_char + len(answers["text"][0]) 
            token_start_index = 0 
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0): 
                token_start_index += 1 
            token_end_index = len(input_ids) - 1 
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0): 
                token_end_index -= 1 
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char): 
                tokenized_examples["start_positions"].append(cls_index) 
                tokenized_examples["end_positions"].append(cls_index) 
            else: 
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char: 
                    token_start_index += 1 
                tokenized_examples["start_positions"].append(token_start_index - 1) 
                while offsets[token_end_index][1] >= end_char: 
                    token_end_index -= 1 
                tokenized_examples["end_positions"].append(token_end_index + 1) 
    return tokenized_examples 


# 准备训练数据并转换为feature
def prepare_train_features(examples):
    tokenized_examples = tokenizer(
        examples["question"],           # 问题文本
        examples["context"],            # 篇章文本
        truncation="only_second",       # 截断只发生在第二部分，即篇章
        max_length=384,                 # 设定最大长度为384
        stride=128,                     # 设定篇章切片步长为128
        return_overflowing_tokens=True, # 返回超出最大长度的标记，将篇章切成多片
        return_offsets_mapping=True,    # 返回偏置信息，用于对齐答案位置
        padding="max_length",           # 按最大长度进行补齐
    )

    # 如果篇章很长，则可能会被切成多个小篇章，需要通过以下函数建立feature到example的映射关系
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # 建立token到原文的字符级映射关系，用于确定答案的开始和结束位置
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # 获取开始和结束位置
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # 获取输入序列的input_ids以及[CLS]标记的位置（在BERT中为第0位）
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # 获取哪些部分是问题，哪些部分是篇章
        sequence_ids = tokenized_examples.sequence_ids(i)

        # 获取答案在文本中的字符级开始和结束位置
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])

        # 获取在当前切片中的开始和结束位置
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        # 检测答案是否超出当前切片的范围
        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            # 超出范围时，答案的开始和结束位置均设置为[CLS]标记的位置
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # 将token_start_index和token_end_index移至答案的两端
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            tokenized_examples["start_positions"].append(token_start_index - 1)
            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples

# 通过函数prepare_train_features，建立分词后的训练集
tokenized_datasets = dataset.map(prepare_train_features, batched=True, remove_columns=dataset["train"].column_names)

# 定义训练参数TrainingArguments，默认使用AdamW优化器
args = TrainingArguments(
    "ft-squad",                         # 输出路径，存放检查点和其他输出文件
    evaluation_strategy="epoch",        # 定义每轮结束后进行评价
    learning_rate=2e-5,                 # 定义初始学习率
    per_device_train_batch_size=16,     # 定义训练批次大小
    per_device_eval_batch_size=16,      # 定义测试批次大小
    num_train_epochs=2,                 # 定义训练轮数
)

# 定义Trainer，指定模型和训练参数，输入训练集、验证集、分词器以及评价函数
trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=default_data_collator,
    tokenizer=tokenizer,
)

# 开始训练！（主流GPU上耗时约几小时）
trainer.train()

