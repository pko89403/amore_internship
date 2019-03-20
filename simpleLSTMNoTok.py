from __future__ import print_function

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import keras
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.preprocessing.text import *
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras import backend as K

training_data = './Input_json/train.json.csv'

df = pd.read_csv(training_data)

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

MAX_SENTENCE_LENGTH = 500
MAX_WORD_COUNT = 50

#X = sequence.pad_sequences( X, maxlen = MAX_SENTENCE_LENGTH )
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)


def RNN(max_len, max_words):
    inputs = Input(name='inputs', shape=[1])
    layer = Embedding(max_words, 64, input_length=max_len, mask_zero = True)(inputs)
    layer = LSTM(64, dropout=0.1, recurrent_dropout=0.1, return_sequences=True)(layer)
    layer = LSTM(64, dropout=0.1, recurrent_dropout=0.1)(layer)
    layer = Dense(128, name = 'FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dense(128, name = 'FC2')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.1)(layer)
    layer = Dense(Y_CLASS, name = 'out_layer')(layer)
    layer = Activation('softmax')(layer)
    model = Model(inputs=inputs, outputs=layer)
    return model


model = RNN(MAX_SENTENCE_LENGTH, MAX_WORD_COUNT)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])

history = model.fit(	X_train,
            		Y_train,
            		batch_size=64,
            		epochs=256,
            		validation_split=0.25,
            		callbacks = [EarlyStopping(monitor='val_loss',
				     patience=16, 
				     min_delta=0.0001)])


score = model.evaluate(X_test, Y_test)

