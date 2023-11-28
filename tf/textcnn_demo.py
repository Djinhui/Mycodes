import pandas as pd
import numpy as np
import jieba
import keras
from keras.layers.merge import concatenate
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dropout, Dense, Input
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn import metrics


#数据预处理
def data_process(path, max_len=50):
    dataset = pd.read_csv(path, sep='\t', names=['text', 'label']).astype(str)
    cw = lambda x: list(jieba.cut(x))
    dataset['words'] = dataset['text'].apply(cw)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(dataset['words'])
    vocab = tokenizer.word_index

    x_train, x_test, y_train, y_test = train_test_split(dataset['words'], dataset['label'], test_size=0.2)
    x_train_word_ids = tokenizer.texts_to_sequences(x_train)
    x_test_word_ids = tokenizer.texts_to_sequences(x_test)
    x_train_padded_seqs = pad_sequences(x_train_word_ids, maxlen=max_len)
    x_test_padded_seqs = pad_sequences(x_test_word_ids, maxlen=max_len)
    return x_train_padded_seqs, y_train, x_test_padded_seqs, y_test, vocab


def TextCNNModel(x_train, y_train, x_test, y_test):
    main_input = Input(shape=(50, ), dtype='float64')
    embedder = Embedding(len(vocab)+1, 300, input_length=50)
    embed = embedder(main_input)
    cnn1 = Conv1D(256, 3, padding='same', strides=1, activation='relu')(embed)
    cnn1 = MaxPooling1D(pool_size=48)(cnn1)
    cnn2 = Conv1D(256, 4, padding='same', strides=1, activation='relu')(embed)
    cnn2 = MaxPooling1D(pool_size=47)(cnn2)
    cnn3 = Conv1D(256, 5, padding='same', strides=1, activation='relu')(embed)
    cnn3 = MaxPooling1D(pool_size=46)(cnn3)

    cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
    flat = Flatten()(cnn) # or use GlobalMaxPooling1D , no need to flatten
    drop = Dropout(0.2)(flat)

    main_output = Dense(3, activation='softmax')(drop)
    model = Model(inputs=main_input, outputs=main_output)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    one_hot_labels = keras.utils.to_categorical(y_train, num_classes=3)
    model.fit(x_train, one_hot_labels, batch_size=32, epochs=30)

    result = model.predict(x_test)
    result_labels = np.argmax(result, axis=1)
    # y_predict = list(max(str, result_labels))
    print('Acc', metrics.accuracy_score(y_test, result_labels))


if __name__ == '__main__':
    path = 'data_train.csv'
    x_train, y_train, x_test, y_test, vocab = data_process(path)
    TextCNNModel(x_train, y_train, x_test, y_test)