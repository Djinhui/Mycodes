# 《自然语言处理实战：从入门到项目实践》CH07 https://github.com/practical-nlp/practical-nlp-code
# 基于文本相似性进行推荐 Recommendation system

# sentence_transformers、flair库 ----> 计算句子向量

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
