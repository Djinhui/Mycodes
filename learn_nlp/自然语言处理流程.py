
# 《自然语言处理实战：从入门到项目实践》CH02 https://github.com/practical-nlp/practical-nlp-code


# 1. 数据获取
# 2. 文本提取和清洗
from bs4 import BeautifulSoup
from urllib.request import urlopen
from pprint import pprint

# from html
myurl = "https://stackoverflow.com/questions/415511/how-to-get-the-current-time-in-python" # specify the url
html = urlopen(myurl).read() # query the website so that it returns a html page  
soupified = BeautifulSoup(html, 'html.parser') # parse the html in the 'html' variable, and store it in Beautiful Soup format

pprint(soupified.prettify()[:20000]) # to get an idea of the hteml structure of the webpage
print(soupified.title)

question = soupified.find('div', {'class':'question'}) # find the necessary tag and class which it belongs to
questiontext = question.find('div', {'class':'s-prose js-post-body'})
print('Question:\n', questiontext.get_text().strip())

answer = soupified.find("div", {"class": "answer"}) # find the nevessary tag and class which it belongs to
answertext = answer.find("div", {"class": "s-prose js-post-body"})
print("Best answer: \n", answertext.get_text().strip())

# from images or pdf
import pytesseract # firstly install tessseract in Windows
from pytesseract import image_to_string
from PIL import Image
import os

# ONLY FOR WINDOWS USERS
# Setting the tesseract path in the script before calling image_to_string.
if (os.name) == "nt":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

image0 = Image.open('OpenSource.png')
image_path = r'OpenSource.png'
extractedIformation = image_to_string(Image.open(image_path))
print(extractedIformation)

# 3. 预处理
#This will be our corpus which we will work on
corpus_original = "Need to finalize the demo corpus which will be used for this notebook and it should be done soon !!. It should be done by the ending of this month. But will it? This notebook has been run 4 times !!"
corpus = "Need to finalize the demo corpus which will be used for this notebook & should be done soon !!. It should be done by the ending of this month. But will it? This notebook has been run 4 times !!"
corpus = corpus.lower()

# removing digits in the corpus
import re
corpus = re.sub(r'\d+', '', corpus)

# removing punctuations
import string
corpus = corpus.translate(str.maketrans('', '', string.punctuation))

# removing trailing whitespaces
corpus = ' '.join([token for token in corpus.split()])

# !python -m spacy download en_core_web_sm

# tokenizing the text
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize,sent_tokenize

stop_words_nltk = set(stopwords.words('english'))
tokenized_corpus_nltk = word_tokenize(corpus)
print("\nNLTK\nTokenized corpus:",tokenized_corpus_nltk)
tokenized_corpus_without_stopwords = [i for i in tokenized_corpus_nltk if not i in stop_words_nltk]
print("Tokenized corpus without stopwords:",tokenized_corpus_without_stopwords)

from spacy.lang.en.stop_words import STOP_WORDS
import spacy

spacy_model = spacy.load('en_core_web_sm')
stopwords_spacy = spacy_model.Defaults.stop_words
print("\nSpacy:")
tokenized_corpus_spacy = word_tokenize(corpus)
print("Tokenized Corpus:",tokenized_corpus_spacy)
tokens_without_sw= [word for word in tokenized_corpus_spacy if not word in stopwords_spacy]

print("Tokenized corpus without stopwords",tokens_without_sw)


print("Difference between NLTK and spaCy output:\n",
      set(tokenized_corpus_without_stopwords)-set(tokens_without_sw))

# Stemming
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

stemmer = PorterStemmer()
print('before stemming:')
print(corpus)

print('after stemming:')
for word in word_tokenize(corpus):
    print(stemmer.stem(word), end=" ")

# Lemmatization
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
for word in word_tokenize(corpus):
    print(lemmatizer.lemmatize(word), end=" ")

# POS Tagging
print('POS Tagging using spacy:')
spacy_model = spacy.load('en_core_web_sm')
doc = spacy_model(corpus_original)
for token in doc:
    print(token, "-->", token.pos_)

for token in doc:
    print(token.text, token.lemma_, token.pos_,
          token.shape_, token.is_alpha, token.is_stop)

print('POS Tagging using nltk')
nltk.download('averaged_perceptron_tagger')
print(nltk.pos_tag(word_tokenize(corpus_original)))

# 4. 特征工程
# 5. 建模
# 6. 评估
# 7. 部署监控