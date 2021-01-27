import itertools
import gensim
from tensorflow.keras.models import Model, Sequential
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from keras.utils.np_utils import to_categorical
from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D, Dropout, Input, Embedding, Dense, Bidirectional
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
np.random.seed(32)

# read the data


def read_files(path):
    file = pd.read_csv(path, sep='\t')
    print('The shape of the data: ', file.shape)
    return file


train_df = read_files('DA_train_labeled.tsv')
dev_df = read_files('DA_dev_labeled.tsv')
test_df = read_files('DA_test_unlabeled.tsv')
train_df

# clean data


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

# prepare Train_X, Dev_X, Test_X

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

# Some varialbles to preprocess the data with keras
# 1) embed_size: the length of each word vector
embed_size = 300
# 2) features: unique words to use
max_features = 50000
# 3) maxlen: max number of words to use
maxlen = 70

# tokenize tweets
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(train_X)

train_sequences = tokenizer.texts_to_sequences(train_X)
dev_sequences = tokenizer.texts_to_sequences(dev_X)
test_sequences = tokenizer.texts_to_sequences(test_X)

X_train = pad_sequences(train_sequences, maxlen=maxlen)
X_dev = pad_sequences(dev_sequences, maxlen=maxlen)
X_test = pad_sequences(test_sequences, maxlen=maxlen)

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

# load the AraVec model for Arabic word embeddings - twitter-CBOW (300 vector size)
print('please wait ... loading the AraVec')
aravec_model = gensim.models.Word2Vec.load(
    './aravec/full_grams_cbow_300_twitter.mdl')
print(aravec_model)
my_dict = {}
for index, key in enumerate(aravec_model.wv.vocab):
    my_dict[key] = aravec_model.wv[key]

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
#embedding_matrix.shape

sequence_input = Input(70,)
embedded_sequences = Embedding(max_features, embed_size, weights=[embedding_matrix],
                               input_length=maxlen,
                               trainable=True)(sequence_input)

x1 = Bidirectional(LSTM(150))(embedded_sequences)
#x2 = Bidirectional(LSTM(150))(x1)
predictions = Dense(N_CLASSES, activation='softmax')(x1)


model = Model(sequence_input, predictions)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

model.fit(X_train, y_train, epochs=3, batch_size=64)

#apply to validation set
pred_dev_y = model.predict([X_dev], batch_size=50, verbose=1)
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
with open("maxlen_60.txt", "w") as f:
    for s in dev_y_predicted:
        f.write(str(s) + "\n")

