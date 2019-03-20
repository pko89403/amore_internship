from __future__ import print_function

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import io
import json

import keras
from keras.utils import plot_model
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.preprocessing.text import *
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras import backend as K
import tensorflowjs as tfjs


training_data = './Input_json/train.json.csv'

df = pd.read_csv(training_data)
print(df.head(), df.info())

sns.countplot(df.cuisine)
plt.xlabel('Label')
plt.title('Number of cusine categories')
plt.savefig('./model_SLSTM/NumberOfCusines.png')
plt.clf()
plt.cla()
plt.close()

# Create input and output Vector
X = df.ingredients
Y = df.cuisine
Y_CLASS = 20 # NUM OF CLASS
# Process the labels

le = LabelEncoder()
Y = le.fit_transform(Y)
Y = Y.reshape(-1,1)
Y = to_categorical(Y, num_classes=Y_CLASS)

# Split into training and test data

TOKEN_MAX_WORDS = 1000
TOKEN_MAX_LEN = 150
def tokenizeData(x_datas):

    """
    :param x_datas:
    Tokenize the data and convert the text to sequences.
    Add padding to ensure that all sequences have the same shape.

    :return: sequences_matrix

    """

    tok = Tokenizer(num_words=TOKEN_MAX_WORDS)
    tok.fit_on_texts(x_datas)
    sequences = tok.texts_to_sequences(x_datas)
    sequences_matrix = sequence.pad_sequences(sequences, maxlen=TOKEN_MAX_LEN)
    
    tok_json = tok.to_json()
    with io.open('tokenizer.json', 'w', encoding='utf-8') as f:
       f.write(json.dumps(tok_json, ensure_ascii=False))




    return sequences_matrix

X = tokenizeData(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)


