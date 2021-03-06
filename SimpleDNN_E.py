from __future__ import print_function

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import io
import json
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Flatten, Dropout, BatchNormalization
from keras.preprocessing.text import *
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import tensorflowjs as tfjs

ODIR = 'DNN_EMB'
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
    
	vocab_size = len(tokenizer.word_index)+1

	global MAX_WORDS, MAX_LEN
	MAX_WORDS = vocab_size
	MAX_LEN = 30
	sequences_matrix = sequence.pad_sequences(sequences, maxlen=MAX_LEN)

	return sequences_matrix

X = tokenizeData(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

def DNN(max_len, max_words):
    inputs = Input(name='inputs', shape=(max_len,))
    embedding = Embedding(input_dim=max_words, output_dim=64, input_length=max_len)(inputs)    

    Flattened = Flatten()(embedding)

    dense1 = Dense(units=2048, activation='relu')(Flattened)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(0.5)(dense1)

    dense2 = Dense(units=2048, activation='relu')(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Dropout(0.5)(dense2)

    dense3 = Dense(units=2048, activation='relu')(dense2)
    dense3 = BatchNormalization()(dense3)
    dense3 = Dropout(0.5)(dense3)

    dense4 = Dense(units=2048, activation='relu')(dense3)
    dense4 = BatchNormalization()(dense4)
    dense4 = Dropout(0.5)(dense4)

    global Y_CLASS
    output = Dense(units=Y_CLASS, activation='softmax')(dense4)
    model = Model(inputs= inputs, outputs = output)

    return model

model = DNN(MAX_LEN, MAX_WORDS)
model.summary()
plot_model(model, to_file='./'+ ODIR + '/model.png')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(	X_train,
			            Y_train,
                        batch_size=200,
                        epochs=1024,
                        validation_split=0.2,
                        callbacks=[EarlyStopping(monitor='val_loss',
                                                 patience=5,
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
