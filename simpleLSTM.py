from __future__ import print_function

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import keras
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.preprocessing.text import Tokenizer
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
plt.show()

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
    return sequences_matrix

X = tokenizeData(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)


def RNN(max_len, max_words):
    inputs = Input(name='inputs', shape=[max_len])
    layer = Embedding(max_words, 128, input_length=max_len)(inputs)
    layer = LSTM(64, dropout=0.1, recurrent_dropout=0.1, return_sequences=True)(layer)
    layer = LSTM(64, dropout=0.1, recurrent_dropout=0.1)(layer)
    layer = Dense(256, name = 'FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dense(256, name = 'FC2')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.05)(layer)
    layer = Dense(Y_CLASS, name = 'out_layer')(layer)
    layer = Activation('softmax')(layer)
    model = Model(inputs=inputs, outputs=layer)
    return model

model = RNN(TOKEN_MAX_LEN, TOKEN_MAX_WORDS)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(  X_train,
            Y_train,
            batch_size=128,
            epochs=256,
            validation_split=0.2,
            callbacks = [EarlyStopping(monitor='val_loss',
				       patience=10, 
				       min_delta=0.0001)])


score = model.evaluate(X_test, Y_test)
print('Test loss: ', score[0])
print('Test Accuracy: ', score[1])

tfjs.converters.save_keras_model(model, "model_SLSTM")

