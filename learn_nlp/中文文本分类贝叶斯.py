import os
import jieba
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

'''
train_path
    enonomy.txt
    fun.txt
    health.txt
    spot.txt
test_path
    enonomy.txt
    fun.txt
    health.txt
    spot.txt

enonomy.txt: 格式label---text
财经---1-10月份全国房产xxxx
财经---10万块钱存银行xxx
...
'''

def load_stopwords(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    words = []
    for line in lines:
        line = line.encode('unicode-escape').decode('unicode-escape')
        line = line.strip()
        words.append(line)
    return words


def load_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    titles = []
    labels = []
    for line in lines:
        line = line.encode('unicode-escape').decode('unicode-escape')

        line = line.strip().rstrip('\n')
        _lines = line.split('---')
        if len(_lines) != 2:
            continue
        label, title = _lines
        title =' '.join(jieba.cut(title))
        title.strip()
        titles.append(title)
        labels.append(label)
    return titles, labels

def load_data(path):
    file_list = os.listdir(path)
    titles_list = []
    labels_list = []
    for filename in file_list:
        file_path = os.path.join(path, filename)
        titles, labels = load_file(file_path)
        titles_list += titles
        labels_list += labels
    return titles_list, labels_list

stop_words = load_stopwords('stopword.txt')
train_datas, train_labels = load_data('train_path')

tf = CountVectorizer(stop_words=stop_words, max_df=0.5)
train_features = tf.fit_transform(train_datas)
clf = MultinomialNB(alpha=0.001).fit(train_features, train_labels)

test_datas, test_labels = load_data('test_path')
test_features = tf.transform(test_datas)
predicted = clf.predict(test_features)
print(metrics.classification_report(test_labels, predicted))


joblib.dump(clf, 'nb.pkl')
joblib.dump(tf, 'tf.pkl')

def load_model(model_path, tf_path):
    model = joblib.load(model_path)
    tf = joblib.load(tf_path)
    return model, tf

def nb_predict(title):
    words = jieba.cut(title)
    s = " ".join(words)
    test_features = tf.transform([s])
    predicted_label = model.predict(test_features)
    return predicted_label[0]

model, tf = load_data('nb.pkl', 'tf.pkl')
nb_predict('留在中超了xxx') # --->运动