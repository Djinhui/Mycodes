# # 《精通Transformer》CH03

# 从零开始针对任何特定语言来训练BERT模型


# 1. 准备原始语料
import pandas as pd
import numpy as np 

imdb_df = pd.read_csv('imdb.csv')
reviews = imdb_df['review'].to_string(index=None)
with open('corpus.txt', 'w') as f:
    f.writelines(reviews)


# 2. 对分词器进行训练
from tokenizers import BertWordPieceTokenizer
bert_wordpiece_tokenizer = BertWordPieceTokenizer()
bert_wordpiece_tokenizer.train('corpus.txt')

# bert_wordpiece_tokenizer.get_vocab() # 训练后词汇表

# !mkdir tokenizer
bert_wordpiece_tokenizer.save_model('tokenizer') # 保存
# tokenizer = BertWordPieceTokenizer.from_file('tokenizer/vocab.txt') # 加载

from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained('tokenizer') # 加载
# tokenizer.mask_token 有些是[MASK]，<mask> 不同模型不同

# 3. 准备语料
from transformers import LineByLineTextDataset, DataCollatorForLanguageModeling
dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path='corpus.txt', block_size=128)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# 4. 配置BERT模型
from transformers import BertConfig, BertForMaskedLM, TrainingArguments, Trainer

# 4.1 训练配置
training_args = TrainingArguments(
    output_dir='./BERT',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=32,
    save_steps=10_000,
    save_total_limit=2,
)

# 4.2 架构配置 

# BERT-BASE (L=12, H=768, A=12, Total Parameters=110M) 
bert = BertForMaskedLM(BertConfig()) # 随机初始化参数， 


#BERT-LARGE (L=24, H=1024, A=16, Total Parameters=340M).
from transformers import BertConfig, BertModel
bert_large= BertConfig(hidden_size=1024, 
                      num_hidden_layers=24 ,
          num_attention_heads=16,
          intermediate_size=4096
     )
model = BertModel(bert_large)

# Albert-base Configuration
from transformers import AlbertConfig, AlbertModel
albert_base = AlbertConfig(
     hidden_size=768,
     num_attention_heads=12,
     intermediate_size=3072,
 )
model = AlbertModel(albert_base)

# ALBERT-xxlarge configuration  by default
from transformers import AlbertConfig, AlbertModel
albert_xxlarge= AlbertConfig()
model = AlbertModel(albert_xxlarge)

# Roberta
from transformers import RobertaConfig, RobertaModel
conf= RobertaConfig()
model = RobertaModel(conf)

# 5. 训练
trainer = Trainer(
    model=bert,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)
trainer.train()

# 6. 保存模型
trainer.save_model('MyBERT')


# 查看模型架构配置
config = BertConfig()
print(config)
'''
BertConfig {
  "attention_probs_dropout_prob": 0.1,
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "transformers_version": "4.8.2",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 30522
}
'''

# 7. 修改模型架构config，创建微型BERT
tiny_bert_config = BertConfig(
    hidden_size=128,
    num_hidden_layers=2,
    num_attention_heads=2,
    intermediate_size=256,
    max_position_embeddings=512,
)

tiny_bert = BertForMaskedLM(tiny_bert_config)

tiny_trainer = Trainer(
    model=tiny_bert,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)
tiny_trainer.train()


# ========================TF加载预训练过的BERT,并将其当作Keras层使用=====================
from transformers import BertTokenizer, TFBertModel

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
bert = TFBertModel.from_pretrained('bert-base-uncased')

print(bert.layers) # TFBertMainLayer中只有一个层，用户可以在Keras模型中访问该层

tokenized_text = tokenizer.batch_encode_plus(["hello how is it going with you","lets test it"], 
                                             return_tensors="tf", max_length=256, truncation=True, pad_to_max_length=True) 
bert(tokenized_text) 

from tensorflow import keras 
import tensorflow as tf 
max_length = 256 
tokens = keras.layers.Input(shape=(max_length,), dtype=tf.dtypes.int32) 
masks = keras.layers.Input(shape=(max_length,), dtype=tf.dtypes.int32) 
embedding_layer = bert.layers[0]([tokens,masks])[0][:,0,:] 
dense = tf.keras.layers.Dense(units=2, activation="softmax")(embedding_layer) 
model = keras.Model([tokens,masks],dense) 

tokenized = tokenizer.batch_encode_plus(["hello how is it going with you","hello how is it going with you"], 
                                        return_tensors="tf", max_length= max_length, truncation=True, pad_to_max_length=True) 
model([tokenized["input_ids"],tokenized["attention_mask"]]) 

model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"]) 
model.summary() 

model.layers[2].trainable = False 


imdb_df = pd.read_csv("IMDB Dataset.csv") 
reviews = list(imdb_df.review) 
tokenized_reviews = tokenizer.batch_encode_plus(reviews, return_tensors="tf", max_length=max_length, truncation=True, pad_to_max_length=True) 


train_split = int(0.8 * len(tokenized_reviews["attention_mask"])) 
train_tokens = tokenized_reviews["input_ids"][:train_split] 
test_tokens = tokenized_reviews["input_ids"][train_split:] 
train_masks = tokenized_reviews["attention_mask"][:train_split] 
test_masks = tokenized_reviews["attention_mask"][train_split:] 
sentiments = list(imdb_df.sentiment) 
labels = np.array([[0,1] if sentiment == "positive" else [1,0] for sentiment in sentiments]) 
train_labels = labels[:train_split] 
test_labels = labels[train_split:] 

model.fit([train_tokens,train_masks],train_labels, epochs=5)