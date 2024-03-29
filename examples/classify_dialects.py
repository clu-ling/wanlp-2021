from typing import Dict, Literal
import logging
import argparse
import pickle
import sys
import re

import numpy as np 
import pandas as pd 

from tqdm                          import tqdm
tqdm.pandas()

import gensim
from gensim.models                 import KeyedVectors
from gensim.test.utils             import datapath

from sklearn.preprocessing         import LabelEncoder
from sklearn.feature_extraction    import DictVectorizer 
from sklearn.base                  import BaseEstimator, TransformerMixin
from sklearn.model_selection       import StratifiedShuffleSplit

import tensorflow as tf
from tensorflow.keras.preprocessing.text     import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers       import Input
from tensorflow.keras.layers       import Conv1D, MaxPooling1D, LSTM, Reshape, Multiply, Bidirectional, Flatten, concatenate, Dropout, Input, Embedding, Dense
from tensorflow.keras.callbacks    import EarlyStopping
from tensorflow.keras.models       import Model, Sequential
from tensorflow.keras.utils        import plot_model
from tensorflow.keras.utils        import to_categorical
from tensorflow.keras.initializers import RandomNormal

# read the data
def load_data(path: str) -> pd.DataFrame:
  """
  Loads dataframes from columnar format
  """
  df = pd.read_csv(
    path, 
    sep="\t" if path.lower().endswith("tsv") else ","
  )
  logging.debug(f"shape: {df.shape}")
  return df

# clean data
def normalize(text: str) -> str:
  normalized = str(text)
  normalized = re.sub('URL','',normalized) # remove links
  normalized = re.sub('USER','',normalized) # remove USER
  normalized = re.sub('#','',normalized) # remove #
  #normalized = re.sub('(@[A-Za-z0-9]+)_[A-Za-z0-9]+','',normalized) # remove @names with underscore
  #normalized = re.sub('(@[A-Za-z0-9]+)','',normalized) # remove @names
  #normalized = re.sub('pic\S+','',normalized) # remove pic.twitter.com links
  normalized = re.sub('\d+','',normalized) # remove numbers
  normalized = re.sub('-','',normalized) # remove symbols - . /
  normalized = re.sub('[a-zA-Z0-9]+','',normalized) # remove English words 
  normalized = re.sub('!','',normalized) # remove English words
  normalized = re.sub(':','',normalized) # remove English words
  normalized = re.sub('[()]','',normalized) # remove English words
  normalized = re.sub('[""]','',normalized) # remove English words
  normalized = re.sub('é','',normalized) # remove English words
  normalized = re.sub('\/','',normalized) # remove English words
  normalized = re.sub('؟','',normalized) # remove English words
  return normalized

# data type alias where value must be 0 or 1
Binary = Literal[0, 1]

class LinguisticFeatureEncoder(DictVectorizer):
  """
  Encodes binary linguistic features
  """
  def __init__(self, **kwargs):        
    super().__init__(sparse=kwargs.get("sparse", False))
    self.use_negative_features: bool = kwargs.get("use_negative_features", True)
    # all positive features
    self.pos_features: Dict[str, Callable[str, Binary]] = kwargs.get(
      "pos_features", 
      {
        # AFRICA
        "egypt_dem": lambda text: 1 if any(text.find(i) >=0 for i in (u'\sدي\s', u'\sده\s', u'\sدى\s')) else 0,
        "egypt_neg": lambda text: 1 if text.find(u'\sمش\s') >= 0 else 0,
        "tunis_iterog": lambda text: 1 if text.find(u'\sعلاش\s') >= 0 else 0,
        "tunis_degree": lambda text: 1 if text.find(u'\sبرشا\s') >= 0 else 0,
        "tunis_contextualword": lambda text: 1 if text.find(u'\sباهي\s') >= 0 else 0,
        "algeria": lambda text: 1 if text.find(u'\sكاش\s') >= 0 else 0,
        "mor_dem": lambda text: 1 if any(text.find(i) >=0 for i in (u'\sديال\s', u'\sديالي\s', u'\sديالى\s')) else 0,
        "mauritania": lambda text: 1 if any(text.find(i) >=0 for i in (u'\sكاغ\s', u'\sايكد\s')) else 0,
        "sudan": lambda text: 1 if text.find(u'\sياخ\s') >= 0 else 0,
        "somalia": lambda text: 1 if text.find(u'\sتناطل\s') >= 0 else 0,
        "dijubuti": lambda text: 1 if any(text.find(i) >=0 for i in (u'\sهاد\s', u'\sهلق\s')) else 0,
        
        # ASIA
        "iraq_degree": lambda text: 1 if any(text.find(i) >=0 for i in (u' خوش ', u' كاعد ')) else 0, 
        "iraq_dem": lambda text: 1 if any(text.find(i) >=0 for i in (u'\sهاي\s', u'\sدا\s')) else 0, 
        "iraq_degree": lambda text: 1 if any(text.find(i) >=0 for i in (u'\sخوش\s', u'\sكاعد\s')) else 0, 
        "iraq_adj": lambda text: 1 if any(text.find(i) >=0 for i in (u'\sفدوه\s', u'\sفدوة\s')) else 0, 
        "iraq_interrog": lambda text: 1 if text.find(u'\sشديحس\s') >= 0 else 0,
        "iraq_tensemarker": lambda text: 1 if any(text.find(i) >=0 for i in (u'\sهسه\s', u'\sهسع\s', u'\sلهسه\s')) else 0, 
        "saudi_dem": lambda text: 1 if text.find(u'\sكذا\s') >= 0 else 0,
        #"qatar": lambda text: 1 if text.find(u'\sوكني\s') >= 0 else 0,
        #"bahrain": lambda text: 1 if text.find(u'\sشفيها\s') >= 0 else 0,
        #"emirates": lambda text: 1 if text.find(u'\sعساه\s') >= 0 else 0,
        #"kuwait": lambda text: 1 if text.find(u'\sعندج\s') >= 0 else 0,
        "oman": lambda text: 1 if text.find(u'\sعيل\s') >= 0 else 0,
        "yemen": lambda text: 1 if text.find(u'\sكدي\s') >= 0 else 0,
        #"syria": lambda text: 1 if text.find(u'\sشنو\s') >= 0 else 0,
        #"palestine": lambda text: 1 if text.find(u'\sليش\s') >= 0 else 0,
        "jordan": lambda text: 1 if text.find(u'\sهاظ\s') >= 0 else 0,
        "lebanon": lambda text: 1 if text.find(u'\sهيدي\s') >= 0 else 0,   
      }
    )
      
    @property
    def size(self) -> int:
      return len(self.get_feature_names())
    
    def create_feature_dict(self, datum) -> Dict[str, Binary]:
      """
      Creates a feature dictionary of str -> 1 or 0.
      Optionally include negated forms of each feature (i.e., NOT_*)
      """
      # 1 if value == 0 else value)
      pos_features = dict((feat, fn(datum)) for (feat, fn) in self.pos_features.items())
      neg_features = dict()
      if not self.use_negative_features:
        return pos_features
      # assumes we're using positive features
      neg_features = dict((f"NOT_{feat}", not value) for (feat, value) in pos_features.items())
      return { **pos_features, **neg_features }
            
    def fit(self, X, y = None):
      dicts = [self.create_feature_dict(datum = datum) for datum in X]
      super().fit(dicts)
        
    def transform(self, X, y = None):
      return super().transform([self.create_feature_dict(datum) for datum in X])

    def fit_transform(self, X, y = None):
      self.fit(X)
      return self.transform(X)

class DialectClassifier(object):
  # FIXME: this should be rewritten to extend BaseEstimator and Predictor (see https://scikit-learn.org/stable/developers/develop.html#instantiation)
  # That would require ... a) the tokenizer, label encoder, etc. to be fit and set as part of .fit(); b) not passing train_df to the constructor; c) passing X and y to .fit() and .predict()
  # if we switch, ... note that "[...] every keyword argument accepted by __init__ should correspond to an attribute on the instance"
  def __init__(self, 
    train_df: pd.DataFrame,
    embeddings_file: str,
    vocab_size: int = 150000,
    # maximum number of tokens in a single doc
    max_seq_len: int = 80,
    x_column: str = "#2_tweet",
    y_column: str = "#3_country_label",
    use_negative_features: bool = True,
  ):
    self.train_df: pd.DataFrame  = train_df
    # column names
    self.x_column: str           = x_column
    self.y_column: str           = y_column
    self.embeddings_file: str    = embeddings_file
    self.VOCAB_SIZE: int         = vocab_size
    self.MAX_LEN: int            = max_seq_len
    self.tokenizer: Tokenizer    = self._create_tokenizer()
    self.w2v, self.embedding_matrix  = self.create_embeddings_matrix()
    self.WORD_EMBEDDING_DIM: int = self.w2v.vector_size
    self.label_encoder: LabelEncoder = self._fit_label_encoder()
    self.NUM_CLASSES: int        = len(self.label_encoder.classes_)
    self.ling_feature_encoder    = self._fit_ling_feature_encoder()
    self.NUM_LING_FEATURES: int  = self.ling_feature_encoder.size
    self.clf: Model              = self.make_classifier()

  def save(self, outfile="dialect-classifier.pkl") -> None:
    pickle.dump(outfile)

  @staticmethod
  def load(model_file: str = "dialect-classifier.pkl") -> "DialectClassifier":
    return pickle.load(model_file)
  
  def _create_tokenizer(self) -> Tokenizer:
    """
    Fits word-piece tokenizer
    """
    tk = Tokenizer(num_words=self.max_features)
    train_X = self.train_df[self.x_column]
    train_X = train_X.astype(str)
    tk.fit_on_texts(train_X)
    #print ('\nTokenizer is working: ', tk)
    return tk
  
  def _transform_text(self, x):
    """
    Tokenizes and pads text
    """
    tk = self.tokenizer
    x = tk.texts_to_sequences(x)
    x = pad_sequences(x, maxlen=maxlen)
    return x
  
  def _fit_ling_feature_encoder(self) -> LinguisticFeatureEncoder:
    """
    Fits and returns a LinguisticFeatureEncoder
    """
    ling_feature_encoder = LinguisticFeatureEncoder(use_negative_features=self.use_negative_features)
    ling_feature_encoder.fit(list(self.train_df[self.x_column].astype(str)))
    return ling_feature_encoder 

  def _transform_ling(self, data):
    """
    Transform raw input to generate linguistic features
    """
    lfe    = self.ling_feature_encoder
    return lfe.transform(l)

  def _fit_label_encoder(self):
    le = LabelEncoder()
    le.fit(self.train_y[self.y_column])
    return le
          
  def _transform_labels(self, y):
    """
    Transform raw input to labels compatible with training
    """
    y = self.label_encoder.transform(y)
    return to_categorical(y, self.N_CLASSES)

  def create_embeddings_matrix(self):
    tk = self.tokenizer
    logging.debug('Loading the word embeddings...')
    w2v = KeyedVectors.load_word2vec_format(self.embeddings_file, binary=True if self.embeddings_file.endswith(".bin") else False, unicode_errors='ignore')
    embed_size: int = w2v.vector_size
    logging.debug(w2v)
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
    #logging.debug(embedding_matrix.shape)
    return w2v, embedding_matrix
      
  def make_classifier(self):
    """
    Intializes a neural network for dialect classification
    """
    ######## component 1 ##########
    inputs_c1     = Input(self.MAX_LEN,)
    embeddings_c1 = Embedding(
      self.VOCAB_SIZE, 
      self.WORD_EMBEDDING_DIM, 
      weights=[self.embedding_matrix],
      input_length=self.MAX_LEN,
      trainable=True
    )(inputs_c1)
    bi_c1         = Bidirectional(LSTM(300))(embeddings_c1)
    drop_c1       = Dropout(0.5)(bi_c1)
    #bidirectional_c1 = Bidirectional(LSTM(300))(drop_c1)
    flat_c1       = Flatten()(drop_c1)

    ######## component 2 ########## 
    inputs_c2       = Input(shape=(self.NUM_LING_FEATURES,))
    embeddings_c2   = Embedding(
      self.NUM_LING_FEATURES, 
      # num. dims
      3,
      embeddings_initializer="uniform",
      embeddings_regularizer=None, 
      activity_regularizer=None,
      embeddings_constraint=None, 
      mask_zero=False, 
      input_length=self.NUM_LING_FEATURES,
      trainable=True
    )(inputs_c2)
    dense_c2_1       = Dense(100, activation="relu")(embeddings_c2)
    drop_c2          = Dropout(0.5)(dense_c2_1)
    dense_c2_2       = Dense(100, activation="relu")(drop_c2)
    flat_c2          = Flatten()(dense_c2_2)
    
    ling_regularizer = tf.Variable(
      tf.random.uniform(
        [flat_c2.shape[-1]], 
        minval=0, 
        maxval=None,
        dtype=tf.dtypes.float32, 
        seed=None, 
        name="ling-regularizer"
      ), 
      trainable=True                             
    )

    ling_regularized = Multiply()([flat_c2, ling_regularizer])

    # merge
    merged           = concatenate([flat_c1, ling_regularized])
    # interpretation
    hidden_1         = Dense(512, activation="relu")(merged)
    drop_1           = Dropout(0.5)(hidden_1)
    hidden_2         = Dense(256, activation="relu")(drop_1)
    outputs          = Dense(self.NUM_CLASSES, activation="softmax")(hidden_2)#"softmax")(hidden_c2_2)
    
    model            = Model(inputs=[inputs_c1, inputs_c2], outputs=outputs)
    # compile
    model.compile(
      loss='categorical_crossentropy', optimizer='adam', 
      metrics=['accuracy']
    )
    # summarize
    #model.summary()
    return model

  def stratify(self,x, y):
    """
    Performs a stratified split of the data
    """
    spiltter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    spiltter.get_n_splits(x, y)
    index    = spiltter.split(x,y)
    for i in index:
      train_index = i[0]
      test_index  = i[1]
      # logging.debug(len(i[0]))
      # logging.debug(len(i[1]))
    return train_index, test_index

  # FIXME: this should be changed to take an X and y (see note preceding class constructor)
  def fit(
    self, 
    max_epochs=10, 
    batch_size=64
  ) -> None:
    """
    Method for training classifier
    """
    clf                 = self.clf
    # preparing for embeddings
    X_text              = self._transform_text(self.train_df[self.x_column])
    y_labels            = self._transform_labels(self.train_df[self.y_column])
    
    train_index, test_index = self.stratify(train_text, train_labels)

    # prepare training for ling
    X_ling              = self._transform_ling(self.train_df[self.x_column].astype(str))
    
    # fit the model
    clf.fit(
      [X_text[train_index], X_ling[train_index]],
      y_labels[train_index],
      validation_data=(
        [X_text[test_index], X_ling[test_index]], 
        y_labels[test_index]
      ),
      epochs=max_epochs, 
      verbose=1, 
      batch_size=batch_size,
      callbacks=[EarlyStopping(monitor='val_loss', patience=2)]
    )
      
  def predict(
    self, 
    X: pd.DataFrame,
    out_file: str,
    batch_size: int = 50
  ):
    """
    Use trained classifier to make predictions
    """
    clf    = self.clf
    X_base = X[self.x_column].astype(str)
    X_text = self._transform_text(X_base)
    X_ling = self._transform_ling(X_base)
    y_hat  = model.predict(
      [X_text, X_ling],
      batch_size=batch_size, 
      verbose=1
    )

    # labels for the predicted dev data
    labels = np.argmax(y_hat, axis=-1)  
    logging.debug(f"Labels are: {labels}")

    # getting the labels(inverse_transform)
    y_predicted = self.label_encoder.inverse_transform(labels)
    logging.debug(f"The length of predicted labels is: {len(y_predicted)}")

    # save labels to txt file
    with open(out_file, "w") as f:
      for s in y_predicted:
        f.write(f"{str(s)}\n") 


if __name__ == "__main__":

  np.random.seed(32)

  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
    "--input", 
    dest="input_file", 
    default="DA_train_labeled.tsv", 
    help="training data file"
  )
  parser.add_argument(
    "--predict", 
    dest="predict_file", 
    type=str,
    default="DA_dev_labeled.tsv", 
    help="data file for which predictions should be generated.  Could correspond to dev, test, or other data."
  )
  parser.add_argument(
    "--embeddings", 
    dest="embeddings_file", 
    type=str,
    default="cbow_100.bin", 
    help="file containing serialized word embeddings (in word2vec format).  Can be binary (.bin) file."
  )
  parser.add_argument(
    "--out", 
    dest="output_file", 
    type=str,
    default="tf.txt", 
    help="write output (predictions) to this file"
  )
  parser.add_argument(
    "--use-neg", 
    dest="use_negative_features", 
    action="store_true",
    help="whether or not to use negative features"
  )
  parser.set_defaults(use_negative_features=False)
  parser.add_argument(
    "--vocab-size", 
    dest="vocab_size", 
    type=int,
    default=150000, 
    help="maximum vocabulary size"
  )
  parser.add_argument(
    "--max-seq-len", 
    dest="max_seq_len", 
    type=int,
    default=80, 
    help="maximum sequence length."
  )
  parser.add_argument(
    "--max-epochs", 
    dest="max_epochs", 
    type=int,
    default=10, 
    help="maximum number of epochs for training"
  )
  parser.add_argument(
    "--batch-size", 
    dest="batch_size", 
    type=int,
    default=50, 
    help="batch size"
  )
  parser.add_argument(
    "-v", "--verbose", 
    dest="verbose", 
    action="store_true",
    help="verbose mode?"
  )
  parser.set_defaults(verbose=False)

  args = parser.parse_args()
  train_file: str             = args.input_file
  # file used to evaluate data.  Could be "dev" or "test"
  predict_file: str           = args.predict_file
  w2v_embeddings_file: str    = args.embeddings_file
  out_file: str               = args.output_file
  # file name to use when serializing trained model
  model_file: str             = "dialect-classifier.pkl"
  x_column: str               = "#2_tweet"
  y_column: str               = "#3_country_label"
  vocab_size: int             = args.vocab_size
  # maximum number of tokens in a single doc
  max_seq_len: int            = args.max_seq_len
  use_negative_features: bool = args.use_negative_features
  max_epochs: int             = args.max_epochs
  batch_size: int             = args.batch_size
  verbose: bool               = args.verbose

  # configure logging
  logging.basicConfig(
    stream=sys.stdout, 
    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG if verbose else logging.INFO
  )

  logging.debug(f"train_file:            {train_file}")
  logging.debug(f"predict_file:          {predict_file}")
  logging.debug(f"x_column:              {x_column}")
  logging.debug(f"y_column:              {y_column}")
  logging.debug(f"out_file:              {out_file}")
  logging.debug(f"use_negative_features: {use_negative_features}")
  logging.debug(f"vocab_size:            {vocab_size}")
  logging.debug(f"max_seq_len:           {max_seq_len}")
  logging.debug(f"max_epochs:            {max_epochs}")
  logging.debug(f"batch_size:            {batch_size}")

  train_df                    = load_data(train_file)
  predict_df                  = load_data(predict_file)

  # normalize text data
  train_df[x_column]    = train_df[x_column].progress_apply(lambda text: normalize(text))
  predict_df[x_column]  = predict_df[x_column].progress_apply(lambda text: normalize(text))

  clf = DialectClassifier(
    train_df=train_df,
    vocab_size=vocab_size,
    max_seq_len=max_seq_len,
    x_column=x_column,
    y_column=y_column,
    use_negative_features=use_negative_features
  )
  # train
  clf.fit(
    max_epochs=max_epochs,
    batch_size=batch_size
  )
  # predict and write to out file 
  clf.predict(
    X=predict_df, 
    out_file=out_file,
    batch_size=batch_size
  )

  # save model
  logging.debug(f"Saving trained model to {model_file}")
  clf.save(model_file)