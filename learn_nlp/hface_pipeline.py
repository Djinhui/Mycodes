from transformers import pipeline

# 分本分类
classifier = pipeline("sentiment-analysis")
print(classifier("I've been waiting for a HuggingFace course my whole life."))
print(classifier("I hate this so much!"))

'''
{'label': 'POSITIVE', 'score': 0.9991129040718079}
{'label': 'NEGATIVE', 'score': 0.9998656511306763}
'''

# 阅读理解
# 问题的答案必须在context中出现过，因为模型的计算过程是从context中找出问题的答案，所以如果问题的答案不在context中，则模型不可能找到答案。/
question_answerer = pipeline("question-answering")
context=r"""
Extractive Question Answering is the task of extracting an answer from a text
given a question. An example of a
question answering dataset is the SQuAD dataset, which is entirely based on
that task. If you would like to fine-tune
a model on a SQuAD task, you may leverage the examples/PyTorch/question-
answering/run_squad.py script.
"""
result = question_answerer(
    question="What is extractive question answering?",
    context=context,
    )
print(result)

result = question_answerer(
    question="What is a good example of a question answering dataset?",
    context=context,
    )
print(result)

'''
{'score': 0.6177279949188232, 'start': 34, 'end': 95, 'answer': 'the task of extracting an answer from a text given a question'}
{'score': 0.5152303576469421, 'start': 148, 'end': 161, 'answer': 'SQuAD dataset'}
'''

# 完形填空

unmasker = pipeline("fill-mask")
sentence = 'HuggingFace is creating a <mask> that the community uses to solve NLP tasks.'
unmasker(sentence)


# 文本生成
text_generator = pipeline("text-generation")
text_generator("In this course, we will teach you all about", max_length=30)

# 命名实体识别
ner = pipeline("ner", grouped_entities=True)
ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")


# 文本摘要
summarizer = pipeline("summarization")
article = """xxxxxx很长的文章"""
summarizer(article, max_length=130, min_length=30, do_sample=False)

# 翻译
translator = pipeline("translation_en_to_de")
translator("Hugging Face is based in New York")

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
translator = pipeline("translation_zh_to_en", model=model, tokenizer=tokenizer)
translator("Hugging Face是一个基于Transformers的深度学习平台")