# # 《精通Transformer》CH04

from tokenizers.models import BPE
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFKC, Sequence, Lowercase
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from tokenizers import ByteLevelBPETokenizer
import tensorflow as tf
import numpy as np


tokenizer = Tokenizer(BPE())
tokenizer.normalizer = Sequence([
    Lowercase()
])
tokenizer.pre_tokenizer = ByteLevel()
tokenizer.decoder = ByteLevelDecoder()


trainer = BpeTrainer(vocab_size=50000, inital_alphabet=ByteLevel.alphabet(), special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>"
        ])
tokenizer.train(["austen-emma.txt"], trainer)


# !mkdir tokenizer_gpt
tokenizer.save("tokenizer_gpt/tokenizer.json")

from transformers import GPT2TokenizerFast, GPT2Config, TFGPT2LMHeadModel

tokenizer_gpt = GPT2TokenizerFast.from_pretrained("tokenizer_gpt")
tokenizer_gpt.add_special_tokens({
  "eos_token": "</s>",
  "bos_token": "<s>",
  "unk_token": "<unk>",
  "pad_token": "<pad>",
  "mask_token": "<mask>"
})

config = GPT2Config(
  vocab_size=tokenizer_gpt.vocab_size,
  bos_token_id=tokenizer_gpt.bos_token_id,
  eos_token_id=tokenizer_gpt.eos_token_id
)
model = TFGPT2LMHeadModel(config)

print(config)

with open("austen-emma.txt", "r", encoding='utf-8') as f:
    content = f.readlines()

content_p = []
for c in content:
    if len(c)>10:
        content_p.append(c.strip())

content_p = " ".join(content_p)+tokenizer_gpt.eos_token

tokenized_content = tokenizer_gpt.encode(content_p)

examples = []
block_size = 100
BATCH_SIZE = 12
BUFFER_SIZE = 1000
for i in range(0, len(tokenized_content)):
    examples.append(tokenized_content[i:i + block_size])

train_data = [] 
labels = [] 
for example in examples: 
    train_data.append(example[:-1]) 
    labels.append(example[1:])


# change 1000 if you want to train on full data
dataset = tf.data.Dataset.from_tensor_slices((train_data[:1000], labels[:1000]))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=[loss, *[None] * model.config.n_layer], metrics=[metric])
# increase number of epochs for higher accuracy and lower loss
num_epoch = 1
history = model.fit(dataset, epochs=num_epoch)


# 使用自回归模型的自然语言生成
def generate(start):  
    input_token_ids = tokenizer_gpt.encode(start, return_tensors='tf')  
    output = model.generate(  
        input_token_ids,  
        max_length = 10,  
        num_beams = 5,  
        temperature = 0.7,  
        no_repeat_ngram_size=2,  
        num_return_sequences=1  
    )  
    return tokenizer_gpt.decode(output[0])

generate("wetson was very good")

# !mkdir my_gpt-2
model.save_pretrained("my_gpt-2/")

model_reloaded = TFGPT2LMHeadModel.from_pretrained("my_gpt-2/")


from transformers import WEIGHTS_NAME, CONFIG_NAME, TF2_WEIGHTS_NAME, AutoModel, AutoTokenizer

tokenizer_gpt.save_pretrained("tokenizer_gpt_auto/")
'''
('tokenizer_gpt_auto/tokenizer_config.json',
 'tokenizer_gpt_auto/special_tokens_map.json',
 'tokenizer_gpt_auto/vocab.json',
 'tokenizer_gpt_auto/merges.txt',
 'tokenizer_gpt_auto/added_tokens.json')
'''
# 文件列表显示在指定的目录中，但必须手动更改tokenizer_config才能使用。
# 首先，应该将其重命名为“config.json”；
# 其次，应该添加一个JavaScript对象表示法(JavaScript Object Notation，JSON)格式的属性，
# 指示model_type属性为gpt2
model = AutoModel.from_pretrained("my_gpt-2/", from_tf = True) 
tokenizer = AutoTokenizer.from_pretrained("tokenizer_gpt_auto")