# 《精通Transformer》CH03

# 1. 加载一个预训练过的tokenizer
from transformers import AutoTokenizer, AutoModel
tokenizerEN = AutoTokenizer.from_pretrained("bert-base-chinese") # 英语分词器
tokenizerTUR = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-uncased",) # 土耳其语言分词器

print(tokenizerEN.vocab_size)
print(tokenizerTUR.vocab_size)
print(tokenizerEN.vocab)



# 2. 从头训练tokenizer
'''
分词管道的过程如下：Normalizer（规范化器）→PreTokenizer（预分词器）→Model Training（模型训练）→Post-processing（后处理）→Decoder（解码器）。

(1)Normalizer（规范化器）：可以使用基本文本处理，如转换为小写、去空格、Unicode规范化和删除重音符号。
(2)PreTokenizer（预分词器）：为下一个训练阶段准备语料库。预分词器根据规则（如空格）将输入拆分为标记。
(3)Model Training（模型训练）：一种子词分词算法（如字节对编码、字节级字节对编码和WordPiece）。
    模型训练发现子词/词汇表，并学习生成规则。
(4)Post-processing（后处理）：提供了与Transformer模型兼容（如BertProcessors）的高级类构造。
    通常在馈入架构之前，向分词输入添加特殊的标记，如［CLS］和［SEP］标记。
(5)Decoder（解码器）：负责将标记ID转换回原始字符串。这只是为了检查正在发生的事情。
'''

# 2.1 训练BPE分词器
import nltk
from nltk.corpus import gutenberg
nltk.download('gutenberg')
nltk.download('punkt')
plays=['shakespeare-macbeth.txt','shakespeare-hamlet.txt','shakespeare-caesar.txt']
shakespeare=[' '.join(s) for ply in plays for s in gutenberg.sents(ply)]

from tokenizers import Tokenizer
from tokenizers.normalizers import (Sequence,Lowercase, NFD, StripAccents)
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.models import BPE
from tokenizers.processors import TemplateProcessing
from tokenizers.decoders import BPEDecoder


special_tokens= ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
temp_proc= TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", special_tokens.index("[CLS]")),
        ("[SEP]", special_tokens.index("[SEP]")),
    ],
)



# Instantiate BPE (Byte-Pair Encoding)
model = BPE()
tokenizer = Tokenizer(model)

# a unicode normalizer, lowercasing and , replacing accents in order  :
# * Sequence : It composes multiple PreTokenizer that will be run in the given order
tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])

# Whitespace: Splits on word boundaries using the regular expression \w+|[^\w\s]+ 
tokenizer.pre_tokenizer = Whitespace() 

tokenizer.post_processor=temp_proc
tokenizer.decoder = BPEDecoder()


from tokenizers.trainers import BpeTrainer
trainer = BpeTrainer(vocab_size=5000, special_tokens= special_tokens)
tokenizer.train_from_iterator(shakespeare, trainer=trainer)
print(f"Trained vocab size: {tokenizer.get_vocab_size()}" )


sen= "Is this a dagger which I see before me, the handle toward my hand?"
sen_enc=tokenizer.encode(sen)
print(f"Output: {format(sen_enc.tokens)}")
# Output: ['[CLS]', 'is', 'this', 'a', 'dagger', 'which', 'i', 'see', 'before', 'me', ',', 'the', 'hand', 'le', 'toward', 'my', 'hand', '?', '[SEP]']

two_enc=tokenizer.encode("I like Hugging Face!","He likes Macbeth!")
print(f"Output: {format(two_enc.tokens)}")
# Output: ['[CLS]', 'i', 'like', 'hu', 'gg', 'ing', 'face', '!', '[SEP]', 'he', 'likes', 'macbeth', '!', '[SEP]']

# 保存并加载整个分词管道
tokenizer.model.save('.') # ['./vocab.json', './merges.txt']

# 保存并加载整个分词管道
tokenizer.save("MyBPETokenizer.json")
tokenizerFromFile=Tokenizer.from_file("MyBPETokenizer.json")
sen_enc3 = tokenizerFromFile.encode("I like HuggingFace and Macbeth")
print(f"Output: {format(sen_enc3.tokens)}")



# 2.1 训练WordPiece分词器
from tokenizers.models import WordPiece
from tokenizers.decoders import WordPiece as WordPieceDecoder
from tokenizers.normalizers import BertNormalizer 

#BERT normalizer includes cleaning the text, handling accents, chinese chars and lowercasing

tokenizer = Tokenizer(WordPiece())
tokenizer.normalizer=BertNormalizer()
tokenizer.pre_tokenizer = Whitespace()

tokenizer.decoder= WordPieceDecoder()


from tokenizers.trainers import WordPieceTrainer
trainer = WordPieceTrainer(vocab_size=5000, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

tokenizer.train_from_iterator(shakespeare, trainer=trainer)
output = tokenizer.encode(sen)
print(output.tokens)
tokenizer.decode(output.ids)



# 3. tokenizers库提供了一个已经制作（未经训练）的空分词管道，其中包含适当的组件，可以构建用于生产的快速原型

'''
# Pre-made tokenizers 
* CharBPETokenizer: The original BPE
* ByteLevelBPETokenizer: The byte level version of the BPE
* SentencePieceBPETokenizer: A BPE implementation compatible with the one used by SentencePiece
* BertWordPieceTokenizer: The famous Bert tokenizer, using WordPiece
'''
from tokenizers import (ByteLevelBPETokenizer,
                            CharBPETokenizer,
                            SentencePieceBPETokenizer,
                            BertWordPieceTokenizer)

tokenizer= SentencePieceBPETokenizer()
print(tokenizer.normalizer)
print(tokenizer.pre_tokenizer)
print(tokenizer.decoder)
print(tokenizer.post_processor)