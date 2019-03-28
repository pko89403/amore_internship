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

MAX_WORDS = 0
MAX_LEN = 0

def tokenizeData(x_datas):

    """
    :param x_datas:
    Tokenize the data and convert the text to sequences.
    Add padding to ensure that all sequences have the same shape.

    :return: sequences_matrix

    """

    #tok = Tokenizer(num_words=TOKEN_MAX_WORDS,char_level=False)
    #tok.fit_on_texts(x_datas)
    #sequences = tok.texts_to_sequences(x_datas)
    #sequences_matrix = sequence.pad_sequences(sequences, maxlen=TOKEN_MAX_LEN)
    
    #tok_json = tok.to_json()
    #with io.open('tokenizer.json', 'w', encoding='utf-8') as f:
    #   f.write(json.dumps(tok_json, ensure_ascii=False))
    #
    #print(tok.texts_to_sequences("red"))

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x_datas)
    sequences = tokenizer.texts_to_sequences(x_datas)
    print(x_datas[0], sequences[0])
    
    maxlen = max([len(x) - 1 for x in sequences])
    vocab_size = len(tokenizer.word_index)+1

    print(maxlen, vocab_size)
    print(tokenizer.texts_to_sequences('romaine lettuce black olives grape tomatoes garlic pepper purple onion seasoning garbanzo beans feta cheese crumbles'))   
    global MAX_WORDS
    MAX_WORDS = vocab_size
    global MAX_LEN
    MAX_LEN = 150
    sequences_matrix = sequence.pad_sequences(sequences, maxlen=MAX_LEN)
    
    tok_json = tokenizer.to_json()
    with io.open('tokenizer2.json', 'w', encoding='utf-8') as f:
       f.write(json.dumps(tok_json, ensure_ascii=False))
    

    return sequences_matrix

X = tokenizeData(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)


def RNN(max_len, max_words):
    print(max_len, max_words)
    inputs = Input(name='inputs', shape=[max_len])
    layer = Embedding(max_words, 128, input_length=max_len)(inputs)
    layer = LSTM(64, dropout=0.1, recurrent_dropout=0.1, return_sequences=True)(layer)
    layer = LSTM(64, dropout=0.1, recurrent_dropout=0.1)(layer)
    layer = Dense(256, name = 'FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dense(256, name = 'FC2')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.1)(layer)
    layer = Dense(Y_CLASS, name = 'out_layer')(layer)
    layer = Activation('softmax')(layer)
    model = Model(inputs=inputs, outputs=layer)
    return model

model = RNN(MAX_LEN, MAX_WORDS)
model.summary()
plot_model(model, to_file='./model_SLSTM/model.png')
model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])

history = model.fit(  X_train,
            Y_train,
            batch_size=64,
            epochs=256,
            validation_split=0.25,
            callbacks = [EarlyStopping(monitor='val_loss',
				       patience=5, 
				       min_delta=0.0001)])

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('./model_SLSTM/train_acc.png')
plt.clf()
plt.cla()
plt.close()
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('./model_SLSTM/train_loss.png')
plt.clf()
plt.cla()
plt.close()

score = model.evaluate(X_test, Y_test)
print('Test loss: ', score[0])
print('Test Accuracy: ', score[1])

tfjs.converters.save_keras_model(model, "model_SLSTM")
