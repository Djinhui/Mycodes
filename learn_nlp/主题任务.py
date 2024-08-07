# 搜索、主题建模、文本摘要、推荐、机器翻译、问答系统
# 《自然语言处理实战：从入门到项目实践》CH07 https://github.com/practical-nlp/practical-nlp-code



# 1. Use Elastic Searech to index and search through data, CMU Book summaries dataset, Dataset Link: https://drive.google.com/drive/u/3/folders/1Hil9ZL7U2hVDF2GaUoDRJgsKO0bq5H7c
from elasticsearch import Elasticsearch
from datetime import datetime

# elastic search instance has to be running on the machine. Default port is 9200.
# Call the Elastic Search instance, and delete any pre-existing index
es=Elasticsearch([{'host':'localhost','port':9200}])
if es.indices.exists(index="myindex"):
    es.indices.delete(index='myindex', ignore=[400, 404]) #Deleting existing index for now

# Build an index from booksummaries dataset. I am using only 500 documents for now.
path = "booksummaries.txt" #Add your path.
count = 1
for line in open(path):
    fields = line.split("\t")
    doc = {'id' : fields[0],
            'title': fields[2],
            'author': fields[3],
            'summary': fields[6]
          }

    res = es.index(index="myindex", id=fields[0], body=doc)
    count = count+1
    if count%100 == 0:
        print("indexed 100 documents")
    if count == 501:
        break

#Check to see how big is the index
res = es.search(index="myindex", body={"query": {"match_all": {}}})
print("Your index has %d entries" % res['hits']['total']['value']) # Your index has 500 entries

#Try a test query. The query searches "summary" field which contains the text
#and does a full text query on that field.
res = es.search(index="myindex", body={"query": {"match": {"summary": "animal"}}})
print("Your search returned %d results." % res['hits']['total']['value']) # Your search returned 16 results.

#Printing the title field and summary field's first 100 characters for 2nd result
print(res["hits"]["hits"][2]["_source"]["title"])
print(res["hits"]["hits"][2]["_source"]["summary"][:100])
'''
Dead Air
 The first person narrative begins on 11 September 2001, and Banks uses the protagonist's conversati
'''

#match query considers both exact matches, and fuzzy matches and works as a OR query. 
#match_phrase looks for exact matches.
while True:
    query = input("Enter your search query: ")
    if query == "STOP":
        break
    res = es.search(index="myindex", body={"query": {"match_phrase": {"summary": query}}})
    print("Your search returned %d results:" % res['hits']['total']['value'])
    for hit in res["hits"]["hits"]:
        print(hit["_source"]["title"])
        #to get a snippet 100 characters before and after the match
        loc = hit["_source"]["summary"].lower().index(query)
        print(hit["_source"]["summary"][:100])
        print(hit["_source"]["summary"][loc-100:loc+100])




# 2. Topic Modeling use LDA and LSA 
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')

from gensim.models import LdaModel
from gensim.corpora import Dictionary

def preprocess(textstring):
    stops = set(stopwords.words('english'))
    tokens = word_tokenize(textstring)
    return [token.lower() for token in tokens if token.isalpha() and token not in stops]

path = "booksummaries.txt" 
summaries = []
for line in open(path,encoding='utf-8'):
    temp = line.split('/t')
    summaries.append(preprocess(temp[6])) # a list of lists of tokens

# Create a dictionary representation of the documents.
dictionary = Dictionary(summaries)

# Filter infrequent or too frequent words.
dictionary.filter_extremes(no_above=0.5, no_below=10)
corpus = [dictionary.doc2bow(summary) for summary in summaries]

# Make a index to word dictionary.
temp = dictionary[0]  # This is only to "load" the dictionary.
id2word = dictionary.id2token

#Train the topic model
model = LdaModel(corpus=corpus, id2word=id2word,iterations=400, num_topics=10)
top_topics = list(model.top_topics(corpus))
print(top_topics)

for idx in range(10):
    print("Topic #%s:" % idx, model.print_topic(idx, 10))
print("=" * 20)
'''
Topic #0: 0.013*"jacky" + 0.006*"dahlia" + 0.005*"novel" + 0.005*"one" + 0.004*"story" + 0.004*"also" + 0.004*"book" + 0.004*"team" + 0.004*"narrator" + 0.003*"jeremy"
Topic #1: 0.010*"book" + 0.009*"war" + 0.006*"in" + 0.006*"world" + 0.005*"novel" + 0.005*"states" + 0.004*"also" + 0.004*"new" + 0.004*"chapter" + 0.004*"story"
Topic #2: 0.008*"he" + 0.007*"she" + 0.006*"mother" + 0.006*"one" + 0.005*"tells" + 0.005*"back" + 0.005*"house" + 0.005*"father" + 0.005*"school" + 0.005*"go"
Topic #3: 0.007*"life" + 0.006*"love" + 0.006*"family" + 0.006*"father" + 0.006*"he" + 0.006*"novel" + 0.005*"young" + 0.005*"she" + 0.004*"story" + 0.004*"one"
Topic #4: 0.007*"he" + 0.006*"one" + 0.004*"murder" + 0.004*"police" + 0.004*"man" + 0.003*"two" + 0.003*"case" + 0.003*"also" + 0.003*"would" + 0.003*"time"
Topic #5: 0.007*"earth" + 0.006*"one" + 0.005*"time" + 0.005*"human" + 0.005*"world" + 0.004*"new" + 0.004*"planet" + 0.004*"life" + 0.003*"space" + 0.003*"he"
Topic #6: 0.006*"he" + 0.005*"they" + 0.005*"one" + 0.004*"back" + 0.004*"find" + 0.004*"ship" + 0.004*"king" + 0.003*"two" + 0.003*"city" + 0.003*"help"
Topic #7: 0.006*"alex" + 0.005*"one" + 0.005*"henry" + 0.004*"freddy" + 0.004*"time" + 0.004*"simon" + 0.004*"he" + 0.004*"luce" + 0.004*"new" + 0.004*"find"
Topic #8: 0.006*"will" + 0.006*"jason" + 0.005*"he" + 0.005*"vampire" + 0.005*"king" + 0.004*"leo" + 0.004*"one" + 0.004*"new" + 0.004*"in" + 0.003*"world"
Topic #9: 0.005*"jake" + 0.005*"charlie" + 0.005*"she" + 0.004*"he" + 0.004*"one" + 0.004*"roger" + 0.004*"back" + 0.004*"they" + 0.004*"luke" + 0.004*"new"
===================
'''

from gensim.models import LsiModel
lsamodel = LsiModel(corpus, num_topics=10, id2word = id2word)  # train model

print(lsamodel.print_topics(num_topics=10, num_words=10))
for idx in range(10):
    print("Topic #%s:" % idx, lsamodel.print_topic(idx, 10))
 
print("=" * 20)
'''
Topic #0: 0.305*"he" + 0.215*"one" + 0.150*"she" + 0.140*"time" + 0.132*"back" + 0.131*"also" + 0.127*"two" + 0.125*"they" + 0.123*"tells" + 0.118*"in"
Topic #1: 0.493*"tom" + 0.226*"sophia" + 0.182*"mrs" + 0.178*"house" + 0.161*"she" + 0.154*"father" + 0.147*"mr" + 0.146*"he" + 0.138*"tells" + -0.126*"one"
Topic #2: -0.558*"tom" + -0.252*"sophia" + 0.213*"she" + 0.190*"he" + -0.185*"mrs" + 0.163*"tells" + 0.144*"mother" + -0.138*"mr" + -0.129*"western" + -0.102*"however"
Topic #3: -0.233*"they" + -0.203*"ship" + 0.187*"he" + -0.183*"david" + -0.178*"back" + -0.165*"tells" + 0.161*"life" + 0.160*"family" + 0.154*"narrator" + -0.154*"find"
Topic #4: 0.664*"he" + -0.258*"mother" + -0.213*"she" + -0.195*"father" + -0.180*"family" + 0.121*"narrator" + 0.120*"monk" + -0.100*"school" + -0.099*"novel" + -0.095*"children"
Topic #5: 0.486*"david" + -0.241*"king" + 0.169*"rosa" + 0.162*"book" + 0.126*"harlan" + -0.120*"he" + 0.111*"she" + 0.108*"gould" + -0.108*"anita" + 0.103*"would"
Topic #6: -0.698*"anita" + -0.471*"richard" + 0.155*"ship" + 0.133*"jacky" + -0.085*"edward" + -0.084*"power" + -0.078*"monk" + 0.073*"father" + -0.071*"scene" + -0.070*"kill"
Topic #7: -0.397*"david" + -0.357*"king" + 0.221*"jacky" + 0.190*"ship" + 0.145*"monk" + 0.134*"doctor" + -0.130*"rosa" + -0.121*"prince" + -0.118*"arthur" + -0.109*"book"
Topic #8: -0.288*"harry" + 0.283*"she" + -0.261*"narrator" + 0.228*"jacky" + -0.224*"david" + -0.195*"monk" + -0.143*"natalie" + 0.130*"ship" + 0.123*"says" + 0.122*"king"
Topic #9: 0.451*"harry" + -0.411*"narrator" + 0.261*"monk" + -0.221*"david" + -0.215*"anita" + 0.182*"natalie" + -0.168*"ship" + -0.163*"he" + -0.153*"richard" + 0.152*"dresden"
====================
'''



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



# 4. recommendation system using doc2vec-----文本相似性
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

data_path = "booksummaries.txt"
mydata = {} #titles-summaries dictionary object
for line in open(data_path, encoding="utf-8"):
    temp = line.split("\t")
    mydata[temp[2]] = temp[6]

train_doc2vec = [TaggedDocument((word_tokenize(mydata[title])), tags=[title]) for title in mydata.keys()]
model = Doc2Vec(vector_size=50, alpha=0.025, min_count=10, dm=1, epochs=100)
model.build_vocab(train_doc2vec)
model.train(train_doc2vec, total_examples=model.corpus_count, epochs=model.epochs)
model.save('d2v.model')

model = Doc2Vec.load('d2v.model')
sample = 'Napoleon enacts changes to the governance structure of the farm, replacing meetings with a committee of pigs who will run the farm.'
new_vector = model.infer_vector(word_tokenize(sample))
sims = model.docvecs.most_similar([new_vector])
print(sims)
# [('Animal Farm', 0.6777619123458862), ('The Wild Irish Girl', 0.6119967699050903), ("Snowball's Chance", 0.60667884349823), ('Family Matters', 0.5831906199455261), ('Settlers in Canada', 0.582908570766449), ('Poor White', 0.5771366953849792), ('The Road to Omaha', 0.576944887638092), ('Ponni', 0.5766265988349915), ("Family Guy: Stewie's Guide to World Domination", 0.5674009323120117), ('Texas Fever', 0.5643234848976135)]
