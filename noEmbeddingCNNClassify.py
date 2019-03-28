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
from keras.layers import Input, Dense, Embedding, Conv1D, MaxPool1D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.optimizers import Adam
from keras.preprocessing.text import *
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras import backend as K
import tensorflowjs as tfjs

ODIR = 'modelCNNwithoutEMB'
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

MAX_WORDS = 0
MAX_LEN = 0

def tokenizeData(x_datas):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(x_datas)
	sequences = tokenizer.texts_to_sequences(x_datas)
    
	maxlen = max([len(x) - 1 for x in sequences])
	vocab_size = len(tokenizer.word_index) + 1
	print(maxlen, vocab_size)

	global MAX_WORDS, MAX_LEN
	MAX_WORDS = vocab_size
	MAX_LEN = 150
	sequences_matrix = sequence.pad_sequences(sequences, maxlen=MAX_LEN)
	print(sequences_matrix[0])

	tok_json = tokenizer.to_json()
	with io.open('./' + ODIR + '/tokenizer_maxLen'+ str(MAX_LEN) + '.json', 'w', encoding='utf-8') as f:
		f.write(json.dumps(tok_json, ensure_ascii=False))
    
	return sequences_matrix

X = tokenizeData(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

def CNN(max_len, max_words):
    inputs = Input(name='inputs', shape=(max_len,))
    Reshaped = Reshape(target_shape=(max_len, 1))(inputs)

    conv_0 = Conv1D(filters=512, kernel_size=2, padding='valid', activation='relu')(Reshaped)
    conv_01 = Conv1D(filters=512, kernel_size=2, padding='valid', activation='relu')(conv_0)
    maxpool_01 = MaxPool1D(pool_size=2, padding='valid')(conv_01)
    conv_012 = Conv1D(filters=512, kernel_size=2, padding='valid', activation='relu')(maxpool_01)
    conv_0123 = Conv1D(filters=512, kernel_size=2, padding='valid', activation='relu')(conv_012)
    maxpool_012 = MaxPool1D(pool_size=2, padding='valid')(conv_0123)

    conv_1 = Conv1D(filters=512, kernel_size=3, padding='valid', activation='relu')(Reshaped)
    conv_11 = Conv1D(filters=512, kernel_size=3, padding='valid', activation='relu')(conv_1)
    maxpool_11 = MaxPool1D(pool_size=2, padding='valid')(conv_11)
    conv_112 = Conv1D(filters=512, kernel_size=3, padding='valid', activation='relu')(maxpool_11)
    conv_1123 = Conv1D(filters=512, kernel_size=3, padding='valid', activation='relu')(conv_112)
    maxpool_112 = MaxPool1D(pool_size=2, padding='valid')(conv_1123)

    conv_2 = Conv1D(filters=512, kernel_size=3, padding='valid', activation='relu')(Reshaped)
    conv_21 = Conv1D(filters=512, kernel_size=3, padding='valid', activation='relu')(conv_2)
    maxpool_21 = MaxPool1D(pool_size=2, padding='valid')(conv_21)
    conv_212 = Conv1D(filters=512, kernel_size=3, padding='valid', activation='relu')(maxpool_21)
    conv_2123 = Conv1D(filters=512, kernel_size=3, padding='valid', activation='relu')(conv_212)
    maxpool_212 = MaxPool1D(pool_size=2, padding='valid')(conv_2123)

    flat_0 = Flatten()(maxpool_012)
    flat_1 = Flatten()(maxpool_112)
    flat_2 = Flatten()(maxpool_212)

    concatenated = Concatenate(axis=1)([flat_0, flat_1, flat_2])

    dense1 = Dense(units=1024, activation='relu')(concatenated)
    dense2 = Dense(units=1024, activation='relu')(dense1)
    dense3 = Dense(units=1024, activation='relu')(dense2)

    global Y_CLASS
    output = Dense(units=Y_CLASS, activation='softmax')(dense3)
    model = Model(inputs=inputs, outputs=output)
	
    return model

model = CNN(MAX_LEN, MAX_WORDS)
model.summary()
plot_model(model, to_file='./'+ ODIR + '/model.png')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(	X_train,
			Y_train,
			batch_size=200,
			epochs=1024,
			validation_split=0.2,
			callbacks = [	EarlyStopping(	monitor='val_loss',
							patience=30,
						)])

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

tfjs.converters.save_keras_model(model, ODIR)