from __future__ import print_function

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

ODIR = 'Transfer_CBOW_LSTM'
TRAINING_DATA_PATH = './Input_json/train.json.csv'

Y_CLASS = 20
LOAD_WORD2VEC_WEIGHT = "./google300Weights.npy"

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

x_limitLen = 30
sequence_matrix = sequence.pad_sequences(sequences, maxlen= x_limitLen)

X = sequence_matrix
Y = transform_reshape_categorical_Y

X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.25,
                                                   )

import numpy as np
embedding_weights = np.load(LOAD_WORD2VEC_WEIGHT)
EMBEDDING_DIM = 300

from keras.models import Model
import keras.backend as K
from keras.layers import Input, Embedding, Reshape, Lambda, LSTM, Dense, Dropout
from keras.utils import plot_model

inputs = Input(shape=(x_limitLen,))
embedding = Embedding(input_dim = x_maxWords,
                      output_dim= 300,
                      weights = [embedding_weights],
                      input_length=x_limitLen,
                      trainable=False)(inputs)
encoder = Lambda(lambda x : K.mean(x, axis=1),
                 output_shape=lambda shape: (shape[0], 1) + shape[2:])( embedding )

reshape = Reshape((1, 300))(encoder)

layer = LSTM(512, return_sequences=True)(reshape)
layer = LSTM(512, return_sequences=True)(layer)
layer = LSTM(512, return_sequences=True)(layer)
layer = LSTM(512, return_sequences=True)(layer)
layer = LSTM(512)(layer)

output = Dense(units=Y_CLASS, activation = 'softmax')(layer)

model = Model(inputs = inputs, output=output)

model.summary()
plot_model(model, to_file='./' + ODIR + '/model.png')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from keras.callbacks import EarlyStopping

history = model.fit(X_train,
                    Y_train,
                    batch_size=200,
                    epochs=256,
                    validation_split=0.2,
                    callbacks = [EarlyStopping(monitor='val_loss',
                                               patience=10,
                                               min_delta=0.001)],
                    verbose=2
                    )

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('./' + ODIR + '/train_acc.png')
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
plt.savefig('./' + ODIR + '/train_loss.png')
plt.clf()
plt.cla()
plt.close()

score = model.evaluate(X_test, Y_test)
print('Test loss: ', score[0])
print('Test Accuracy: ', score[1])

import tensorflowjs as tfjs
tfjs.converters.save_keras_model(model, ODIR)