# 《精通Transformer》CH06   See At finetune_bert_src.py 
# 输出每个token是答案起始索引和终止索引的概率

from datasets import load_dataset 
squad = load_dataset("squad_v2") 
print(squad)
'''
DatasetDict({
    train: Dataset({
        features: ['answers', 'context', 'id', 'question', 'title'],
        num_rows: 130319
    })
    validation: Dataset({
        features: ['answers', 'context', 'id', 'question', 'title'],
        num_rows: 11873
    })
})
'''

from transformers import AutoTokenizer 
model = "distilbert-base-uncased" 
tokenizer = AutoTokenizer.from_pretrained(model) 

max_length = 384 
doc_stride = 128 
example = squad["train"][173] 
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

"""
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
        max_length=max_length, 
        stride=doc_stride, 
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


tokenized_datasets = squad.map(prepare_train_features, batched=True, remove_columns=squad["train"].column_names)

from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer 
from transformers import default_data_collator 

data_collator = default_data_collator 

model = AutoModelForQuestionAnswering.from_pretrained(model) 

args = TrainingArguments( 
f"test-squad", 
evaluation_strategy = "epoch", 
learning_rate=2e-5, 
per_device_train_batch_size=16, 
per_device_eval_batch_size=16, 
num_train_epochs=3, 
weight_decay=0.01, 
)

trainer = Trainer( 
model, 
args, 
train_dataset=tokenized_datasets["train"], 
eval_dataset=tokenized_datasets["validation"], 
data_collator=data_collator, 
tokenizer=tokenizer, 
) 


trainer.train() 

trainer.save_model("distillBERT_SQUAD")

from transformers import pipeline 
qa_model = pipeline('question-answering', model='distilbert-base-cased-distilled-squad', tokenizer='distilbert-base-cased') 

question = squad["validation"][0]["question"] 
context = squad["validation"][0]["context"] 
print("Question:") 
print(question) 
print("Context:") 
print(context) 

qa_model(question=question, context=context) 