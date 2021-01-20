from sklearn.model_selection import StratifiedShuffleSplit
import itertools
import numpy as np 
import pandas as pd 
from tqdm import tqdm
tqdm.pandas()
import re
import matplotlib.pyplot as plt
np.random.seed(32)

import gensim
from gensim.models import KeyedVectors
from gensim.test.utils import datapath

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D,Flatten, concatenate, Dropout, Input, Embedding, Dense, Bidirectional
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model, Sequential
from keras.callbacks import EarlyStopping
from sklearn.feature_extraction import DictVectorizer 

########################### configurations ####################
# 1) embed_size: the length of each word vector
embed_size = 300
# 2) features: unique words to use
max_features = 50000
# 3) maxlen: max number of words to use
maxlen = 100
# the number of samples to use for one update
batch = 64
# the max number of epochs to use
num_epochs = 2
# data
train = 'DA_train_labeled.tsv'
dev = 'DA_dev_labeled.tsv'
test = 'DA_test_unlabeled.tsv'
w2v_data = 'cbow_100.bin'
ling_size = 29

########################### read the data ####################
def read_files(path):
    file = pd.read_csv(path, sep='\t')
    print('shape', file.shape)
    return file

train_df = read_files(train)
dev_df = read_files(dev)
test_df = read_files(test)

########################### clean the data ####################
def normalize(text):
    normalized = str(text)
    normalized = re.sub('URL', '', normalized)  # remove links
    normalized = re.sub('USER', '', normalized)  # remove USER
    normalized = re.sub('#', '', normalized)  # remove #
    #normalized = re.sub('(@[A-Za-z0-9]+)_[A-Za-z0-9]+','',normalized) # remove @names with underscore
    #normalized = re.sub('(@[A-Za-z0-9]+)','',normalized) # remove @names
    #normalized = re.sub('pic\S+','',normalized) # remove pic.twitter.com links
    normalized = re.sub('\d+', '', normalized)  # remove numbers
    normalized = re.sub('-', '', normalized)  # remove symbols - . /
    normalized = re.sub('[a-zA-Z0-9]+', '', normalized)  # remove English words
    normalized = re.sub('!', '', normalized)  # remove English words
    normalized = re.sub(':', '', normalized)  # remove English words
    normalized = re.sub('[()]', '', normalized)  # remove English words
    normalized = re.sub('☻', '', normalized)  # remove English words
    normalized = re.sub('[""]', '', normalized)  # remove English words
    normalized = re.sub('é', '', normalized)  # remove English words
    normalized = re.sub('\/', '', normalized)  # remove English words
    normalized = re.sub('؟', '', normalized)  # remove English words
    return normalized


train_df['#2_tweet'] = train_df['#2_tweet'].progress_apply(
    lambda text: normalize(text))
dev_df['#2_tweet'] = dev_df['#2_tweet'].progress_apply(
    lambda text: normalize(text))
test_df['#2_tweet'] = test_df['#2_tweet'].progress_apply(
    lambda text: normalize(text))

########################### DA Class ####################


class DA(object):
    def __init__(self):
        self.tokenizer = self.tokenize()
        self.w2v, self.embedding_matrix = self.create_embeddings_matrix()
        self.encoder = LabelEncoder()
        self.net = self.network()

    def tokenize(self):
        tk = Tokenizer(num_words=max_features)
        train_X = train_df["#2_tweet"]
        train_X = train_X.astype(str)
        tk.fit_on_texts(train_X)
        print('\ntokenizer is working: ', tk)
        return tk

    def prepare_text(self, x):
        tk = self.tokenizer
        x = tk.texts_to_sequences(x)
        x = pad_sequences(x, maxlen=maxlen)
        return x

    def prepare_labels(self, y):
        self.encoder.fit(y)
        y = self.encoder.transform(y)
        N_CLASSES = np.max(y) + 1
        y = to_categorical(y, N_CLASSES)
        print('Shape of label tensor:', y.shape)
        return y

    def create_embeddings_matrix(self):
        tk = self.tokenizer
        print('please wait ... loading the word embeddings')
        w2v = KeyedVectors.load_word2vec_format(
            w2v_data, binary=True, unicode_errors='ignore')
        print(w2v)
        my_dict = {}
        for index, key in enumerate(w2v.wv.vocab):
            my_dict[key] = w2v.wv[key]
        embedding_matrix = np.zeros((max_features, embed_size))
        for word, index in tk.word_index.items():
            if index > max_features - 1:
                break
            else:
                embedding_vector = my_dict.get(word)
                if embedding_vector is not None:
                    embedding_matrix[index] = embedding_vector
        print(embedding_matrix.shape)
        return w2v, embedding_matrix

    def network(self):
        model = Sequential()
        model.add(Embedding(max_features,
                            embed_size,
                            weights=[self.embedding_matrix],
                            input_length=maxlen,
                            trainable=True))
        #model.add(Bidirectional(LSTM(300, return_sequences=True)))
        model.add(Bidirectional(LSTM(300)))
        model.add(Dense(21, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['acc'])
        #print(model.summary())
        return model

    def train(self):
        model = self.net
        train_text = self.prepare_text(train_df['#2_tweet'])
        train_labels = self.prepare_labels(train_df['#3_country_label'])

        model.fit(train_text, train_labels, epochs=num_epochs,
                  verbose=1, batch_size=batch)

    def predict_dev(self):
        model = self.net
        dev_X = dev_df["#2_tweet"]
        dev_X = dev_X.astype(str)
        dev_text = self.prepare_text(dev_X)
        pred_dev_y = model.predict([dev_text], batch_size=50, verbose=1)

        # labels for the predicted dev data
        labels = np.argmax(pred_dev_y, axis=-1)
        print('Labels are: ', labels)

        # getting the labels(inverse_transform)
        dev_y_predicted = self.encoder.inverse_transform(labels)
        print('The length of predicted labels is: ', len(dev_y_predicted))

        # save labels to txt file
        with open("maxlen_60_2_epochs.txt", "w") as f:
            for s in dev_y_predicted:
                f.write(str(s) + "\n")


if __name__ == "__main__":
    da = DA()
    da.train()
    da.predict_dev()