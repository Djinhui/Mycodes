# 《自然语言处理实战：从入门到项目实践》CH07 https://github.com/practical-nlp/practical-nlp-code

# 3. Text Summarization
# 3.1 Summarization with Sumy

import nltk
from sumy.parsers.html import HtmlParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.lsa import LsaSummarizer

num_sentences_in_summary = 2
url = "https://en.wikipedia.org/wiki/Automatic_summarization"
parser = HtmlParser.from_url(url, Tokenizer("english"))

summarizer_list=("TextRankSummarizer:","LexRankSummarizer:","LuhnSummarizer:","LsaSummarizer") #list of summarizers
summarizers = [TextRankSummarizer(), LexRankSummarizer(), LuhnSummarizer(), LsaSummarizer()]

for i,summarizer in enumerate(summarizers):
    print(summarizer_list[i])
    for sentence in summarizer(parser.document, num_sentences_in_summary):
        print((sentence))
    print("-"*30)


# 3.2 Summarization with Gensim
from gensim.summarization import summarize,summarize_corpus
from gensim.summarization.textcleaner import split_sentences
from gensim import corpora

text = open("nlphistory.txt").read()

#summarize method extracts the most relevant sentences in a text
print("Summarize:\n",summarize(text, word_count=200, ratio = 0.1))


#the summarize_corpus selects the most important documents in a corpus:
sentences = split_sentences(text)# Creates a corpus where each document is a sentence.
tokens = [sentence.split() for sentence in sentences]
dictionary = corpora.Dictionary(tokens)
corpus = [dictionary.doc2bow(sentence_tokens) for sentence_tokens in tokens]

# Extracts the most important documents (shown here in BoW representation)
print("-"*30,"\nSummarize Corpus\n",summarize_corpus(corpus,ratio=0.1))


# 3.2 Summarization with Summa
from summa import summarizer
from summa import keywords
text = open("nlphistory.txt").read()

print("Summary:")
print (summarizer.summarize(text,ratio=0.1))


# 3.3 BERT for Extractive Summarization
'''
#Install the required libraries
!pip install bert-extractive-summarizer
!pip install spacy==2.1.3
!pip install transformers==2.2.2
!pip install neuralcoref
!pip install torch #you can comment this line if u already have tensorflow2.0 installed
!pip install neuralcoref --no-binary neuralcoref
!python -m spacy download en_core_web_sm
'''
#sowyma could you please look at this coreference vs without coreference. I personally think we need to use a better input.
#currently using the same one as above the nlphistory.txt

from summarizer import Summarizer
from summarizer.coreference_handler import CoreferenceHandler

model = Summarizer()

print("Without Coreference:")
result = model(text, min_length=200,ratio=0.01)
full = ''.join(result)
print(full)


print("With Coreference:")
handler = CoreferenceHandler(greedyness=.35)

model = Summarizer(sentence_handler=handler)
result = model(text, min_length=200,ratio=0.01)
full = ''.join(result)
print(full)


# 3.4 BERT for Abstractive Summarization
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

device = torch.device('cpu')

text ="""
don’t build your own MT system if you don’t have to. It is more practical to make use of the translation APIs. When we use such APIs, it is important to pay closer attention to pricing policies. It would perhaps make sense to store the translations of frequently used text (called a translation memory or a translation cache). 

If you’re working with a entirely new language, or say a new domain where existing translation APIs do poorly, it would make sense to start with a domain knowledge based rule based translation system addressing the restricted scenario you deal with. Another approach to address such data scarce scenarios is to augment your training data by doing “back translation”. Let us say we want to translate from English to Navajo language. English is a popular language for MT, but Navajo is not. We do have a few examples of English-Navajo translation. In such a case, one can build a first MT model between Navajo-English, and use this system to translate a few Navajo sentences into English. At this point, these machine translated Navajo-English pairs can be added as additional training data to English-Navajo MT system. This results in a translation system with more examples to train on (even though some of these examples are synthetic). In general, though, if accuracy of translation is paramount, it would perhaps make sense to form a hybrid MT system which combines the neural models with rules and some form of post-processing, though. 

"""


preprocess_text = text.strip().replace("\n","")
t5_prepared_Text = "summarize: "+preprocess_text
print ("original text preprocessed: \n", preprocess_text)

tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)


# summmarize 
summary_ids = model.generate(tokenized_text,
                                    num_beams=4,
                                    no_repeat_ngram_size=2,
                                    min_length=30,
                                    max_length=100,
                                    early_stopping=True)
#there are more parameters which can be found at https://huggingface.co/transformers/model_doc/t5.html

output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print ("\n\nSummarized text: \n",output)