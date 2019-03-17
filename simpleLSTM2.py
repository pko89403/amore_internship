from __future__ import print_function

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import keras
from keras.models import Sequential
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras import backend as K
import tensorflowjs as tfjs


training_data = './Input_json/train.json.csv'

df = pd.read_csv(training_data)

# Create input and output Vector
X = df.ingredients
Y = df.cuisine
Y_CLASS = 20
# Process the labels

le = LabelEncoder()
Y = le.fit_transform(Y)
Y = Y.reshape(-1,1)
Y = to_categorical(Y, num_classes=Y_CLASS)


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
    #print(tok.word_index)
    return sequences_matrix



X = tokenizeData(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

print('Data Example : ', X_train[0], Y_train[0])
print((X_train.shape, Y_train.shape, X_test.shape, Y_test.shape))


model = Sequential()
emb_dim = 128
model.add(Embedding(TOKEN_MAX_WORDS, 128, input_length=X.shape[1]))
model.add(LSTM(64, dropout=0.7, recurrent_dropout=0.7))
model.add(Dense(Y_CLASS, activation='softmax'))

model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
print(model.summary())
history = model.fit(X_train,
		    Y_train, 
		    epochs=10, 
                    batch_size=256,
                    validation_split=0.2,
                    callbacks=[EarlyStopping(monitor='val_loss',
                    patience=7, 
                    min_delta=0.0001)])

score = model.evaluate(X_test, Y_test)
print('Test loss: ', score[0])
print('Test Accuracy: ', score[1])

tfjs.converters.save_keras_model(model, "model_SLSTM")

