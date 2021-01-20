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

########################### Linguistic features ####################


class FeatureExtractor(object):
    # AFRICA: Egypt
    def egypt_neg(self, text):
        return 1 if u'مش' in text else 0

    def egypt_dem(self, text):
        return 1 if (u'ده' or u'دي' or 'دى') in text else 0
    # AFRICA: Libya

    def libya(self, text):
        return 1 if u'شن' in text else 0
    # AFRICA: Tunisia

    def tunis(self, text):
        return 1 if u'يطيشوه' in text else 0

    def tunis_iterog(self, text):
        return 1 if u'علاش' in text else 0

    def tunis_degree(slef, text):
        return 1 if u'برشا' in text else 0

    def tunis_contextualword(self, text):
        return 1 if u'باهي' in text else 0
    # AFRICA: Algeria

    def algeria(self, text):
        return 1 if u'كاش' in text else 0
    # AFRICA: Moroccoa

    def mor_dem(self, text):
        return 1 if (u'ديال' or u'ديالي' or 'ديالى') in text else 0
    # AFRICA: Mauritania

    def mauritania(self, text):
        return 1 if (u'كاغ' or u'ايكد') in text else 0
    # AFRICA: Sudan

    def sudan(self, text):
        return 1 if u'ياخ' in text else 0
    # AFRICA: Di

    def dijubuti(self, text):
        return 1 if (u'هاد' or u'هلق') in text else 0
    # AFRICA: Somalia

    def somalia(self, text):
        return 1 if u'تناطل' in text else 0

    ##########################
    # ASIA: Iraq
    def iraq_degree(self, text):
        return 1 if (u'خوش' or u'كاعد') in text else 0

    def iraq_dem(self, text):
        return 1 if (u'هاي' or u'دا') in text else 0

    def iraq_adj(self, text):
        return 1 if (u'فدوه' or u'فدوة') in text else 0

    def iraq_interrog(self, text):
        return 1 if u'شديحس' in text else 0

    def iraq_tensemarker(self, text):
        return 1 if (u'هسه' or u'هسع' or u'لهسه') in text else 0
    # ASIA: Saudi

    def saudi_dem(self, text):
        return 1 if u'كذا' in text else 0
    # ASIA: Qatar

    def qatar(self, text):
        return 1 if u'وكني' in text else 0
    # ASIA: Bahrain

    def bahrain(self, text):
        return 1 if u'شفيها' in text else 0
    # ASIA: UAE

    def emirates(self, text):
        return 1 if u'عساه' in text else 0
    # ASIA: Kuwait

    def kuwait(self, text):
        return 1 if u'عندج' in text else 0
    # ASIA: Oman

    def oman(self, text):
        return 1 if u'عيل' in text else 0
    # ASIA: Yemen

    def yemen(self, text):
        return 1 if u'كدي' in text else 0
    # ASIA: Syria

    def syria(self, text):
        return 1 if u'شنو' in text else 0
    # ASIA: Palestine

    def palestine(self, text):
        return 1 if u'ليش' in text else 0
    # ASIA: Jordan

    def jordan(self, text):
        return 1 if u'هاظ' in text else 0
    # ASIA: Lebanon

    def lebanon(self, text):
        return 1 if u'هيدي' in text else 0

    # create feature dictionary collects the previous function

    def create_feature_dict(self, text):
        return{
            "egypt_neg": self.egypt_neg(text),
            "egypt_dem": self.egypt_dem(text),
            "libya": self.libya(text),
            "tunis": self.tunis(text),
            "tunis_iterog": self.tunis_iterog(text),
            "tunis_degree": self.tunis_degree(text),
            "tunis_contextualword": self.tunis_contextualword(text),
            "algeria": self.algeria(text),
            "mor_dem": self.mor_dem(text),
            "mauritania": self.mauritania(text),
            "sudan": self.sudan(text),
            "dijubuti": self.dijubuti(text),
            "somalia": self.somalia(text),
            "iraq_degree": self.iraq_degree(text),
            "iraq_dem": self.iraq_dem(text),
            "iraq_adj": self.iraq_adj(text),
            "iraq_interrog": self.iraq_interrog(text),
            "iraq_tensemarker": self.iraq_tensemarker(text),
            "saudi_dem": self.saudi_dem(text),
            "qatar": self.qatar(text),
            "bahrain": self.bahrain(text),
            "emirates": self.emirates(text),
            "kuwait": self.kuwait(text),
            "oman": self.oman(text),
            "yemen": self.yemen(text),
            "syria": self.syria(text),
            "palestine": self.palestine(text),
            "jordan": self.jordan(text),
            "lebanon": self.lebanon(text)
        }


class FeatureEncoder(object):
    def __init__(self):
        self.dv = DictVectorizer()

    def fit_and_transform(self, texts):
        m = self.dv.fit_transform(texts)
        return m

    def get_feature_names(self):
        feature_names = self.dv.get_feature_names()
        return feature_names

def linguistic_features(data):
    feature_extraction = FeatureExtractor()
    feature_reps = []
    for sent in data:
        feature_dict = feature_extraction.create_feature_dict(sent)

        feature_reps.append(list(feature_dict.values()))

    linguistic_vector = np.array(feature_reps)
    return linguistic_vector


########################### Tokenizer and padding sequences ####################
# train X, val X, test X
train_X = train_df["#2_tweet"]
dev_X = dev_df["#2_tweet"]
test_X = test_df["#2_tweet"]

# target values
train_y = train_df['#3_country_label']
#print (train_y)
dev_y = dev_df['#3_country_label']
#print (dev_y)

train_X = train_X.astype(str)
dev_X = dev_X.astype(str)
test_X = test_X.astype(str)

# tokenize tweets
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(train_X)

train_sequences = tokenizer.texts_to_sequences(train_X)
dev_sequences = tokenizer.texts_to_sequences(dev_X)
test_sequences = tokenizer.texts_to_sequences(test_X)

X_train = pad_sequences(train_sequences, maxlen=maxlen)
X_dev = pad_sequences(dev_sequences, maxlen=maxlen)
X_test = pad_sequences(test_sequences, maxlen=maxlen)

train_ling = linguistic_features(train_X)
dev_ling = linguistic_features(dev_X)
test_ling = linguistic_features(test_X)

# encode y data labels

encoder = LabelEncoder()
encoder.fit(train_y)
y_train = encoder.transform(train_y)
y_dev = encoder.transform(dev_y)

N_CLASSES = np.max(y_train) + 1
N_CLASSES
y_train = to_categorical(y_train, N_CLASSES)
y_dev = to_categorical(y_dev, N_CLASSES)
print('Shape of label tensor:', y_train.shape)

########################### word embeddings ####################

# load the AraVec model for Arabic word embeddings - twitter-CBOW (300 vector size)
print('please wait ... loading the AraVec')
# load w2v model
w2v_vectors_file = './cbow_100.bin'
w2v = KeyedVectors.load_word2vec_format(
    w2v_vectors_file, binary=True, unicode_errors='ignore')
print(w2v)

my_dict = {}
for index, key in enumerate(w2v.wv.vocab):
    my_dict[key] = w2v.wv[key]

embedding_matrix = np.zeros((50000, 300))
for word, index in tokenizer.word_index.items():
    if index > 50000 - 1:
        break
    else:
        embedding_vector = my_dict.get(word)
        #print (embedding_vector)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
            #print (len(embedding_matrix[index]))
embedding_matrix.shape

########################### K-folds ####################

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
sss.get_n_splits(X_train, y_train)

# for train_index, test_index in sss.split(X, y):
# ...     print("TRAIN:", train_index, "TEST:", test_index)
# ...     X_train, X_test = X[train_index], X[test_index]
# ...     y_train, y_test = y[train_index], y[test_index]
# TRAIN: [5 2 3] TEST: [4 1 0]
# TRAIN: [5 1 4] TEST: [0 2 3]
# TRAIN: [5 0 2] TEST: [4 3 1]
# TRAIN: [4 1 0] TEST: [2 3 5]
# TRAIN: [0 5 1] TEST: [3 4 2]
index = sss.split(X_train, y_train)
for x in index:
    train_index = x[0]
    test_index = x[1]
    print(len(x[0]))
    print(len(x[1]))

########################### The model ####################
# the cnn model


def cnn(length=maxlen, vocab_size=max_features):
    # channel 1
    inputs1 = Input(maxlen,)
    embedding1 = Embedding(max_features, embed_size, weights=[embedding_matrix],
                           input_length=maxlen,
                           trainable=True)(inputs1)
    bidirectional1 = Bidirectional(
        LSTM(300, return_sequences=True))(embedding1)
    drop1 = Dropout(0.5)(bidirectional1)
    bidirectional2 = Bidirectional(LSTM(300))(drop1)
    flat1 = Flatten()(bidirectional2)
    # channel 2
    inputs2 = Input(ling_size,)
    dense = Dense(32, activation='relu')(inputs2)
    drop2 = Dropout(0.5)(dense)

    # merge
    merged = concatenate([flat1, drop2])
    # interpretation
    dense1 = Dense(332, activation='relu')(merged)
    outputs = Dense(21, activation='softmax')(dense1)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    # compile
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    # summarize
    model.summary()
    return model


model = cnn()

########################### Train ####################

train = model.fit([X_train[train_index], train_ling[train_index]], y_train[train_index],
                  validation_data=(
                      [X_train[test_index], train_ling[test_index]], y_train[test_index]),
                  epochs=10, batch_size=32, callbacks=[EarlyStopping(monitor='val_loss', patience=2)])

########################### apply on validation ####################
#apply to validation set
pred_dev_y = model.predict([X_dev, dev_ling], batch_size=50, verbose=1)
pred_dev_y
indexes = np.argsort(pred_dev_y)[::-1]
indexes

# labels for the predicted dev data
labels = np.argmax(pred_dev_y, axis=-1)
print('Labels are: ', labels)

# getting the labels throw (inverse_transform)
dev_y_predicted = encoder.inverse_transform(labels)
print('The length of predicted labels is: ', len(dev_y_predicted))

# save labels to txt file
with open("two_forks_early.txt", "w") as f:
    for s in dev_y_predicted:
        f.write(str(s) + "\n")
