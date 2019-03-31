import pandas as pd

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split


ODIR = 'PretrainedWithCNN'
TRAINING_DATA_PATH = './Input_json/train.json.csv'

Y_CLASS = 20
LOAD_WORD2VEC_PATH = '../pretrained_emb_weights/GoogleNews-vectors-negative300.bin.gz'



df = pd.read_csv(TRAINING_DATA_PATH)

X = df.ingredients
Y = df.cuisine

le = LabelEncoder()
transform_Y = le.fit_transform(Y)
transform_reshape_Y = transform_Y.reshape(-1, 1)
transform_reshape_categorical_Y = to_categorical(transform_reshape_Y,
                                                 num_classes=Y_CLASS)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)

x_maxLen = max([len(x) - 1 for x in sequences])
x_maxWords = len(tokenizer.word_index) + 1

print('Tokenizing Result', x_maxLen, ', ', x_maxWords)
x_limitLen = 30
sequence_matrix = sequence.pad_sequences(sequences, maxlen= x_limitLen)

X = sequence_matrix
Y = transform_reshape_categorical_Y

X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.25,
                                                    random_state=42)

import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.models.keyedvectors import KeyedVectors

word_vectors = KeyedVectors.load_word2vec_format(LOAD_WORD2VEC_PATH,
                                                 binary=True)

import numpy as np

EMBEDDING_DIM = 300
vocabulary_size = x_maxWords

embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))
for word, i in tokenizer.word_index.items():
    if( i >= vocabulary_size):
        continue
    try:
        embedding_vector = word_vectors[word]
        embedding_matrix[i] = embedding_vector
    except KeyError:
        embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25), EMBEDDING_DIM)

del(word_vectors)

np.save("./google300Weights", embedding_matrix)