from __future__ import print_function

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from keras.utils import plot_model
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Input, Embedding
from keras.preprocessing.text import *
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import tensorflowjs as tfjs

ODIR = 'SELF_EMB_LSTM'
TRAINING_DATA_PATH = './Input_json/train.json.csv'

df = pd.read_csv(TRAINING_DATA_PATH)
X = df.ingredients
Y = df.cuisine
Y_CLASS = 20 # NUM OF CLASS

le = LabelEncoder()
Y = le.fit_transform(Y)
Y = Y.reshape(-1,1)
Y = to_categorical(Y, num_classes=Y_CLASS)

MAX_WORDS = 0
MAX_LEN = 30

def tokenizeData(x_datas):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x_datas)
    sequences = tokenizer.texts_to_sequences(x_datas)

    vocab_size = len(tokenizer.word_index)+1

    global MAX_WORDS, MAX_LEN
    MAX_WORDS = vocab_size
    sequences_matrix = sequence.pad_sequences(sequences, maxlen=MAX_LEN)
    return sequences_matrix

X = tokenizeData(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

def LSTM_Model(max_len, max_words):
    print(max_len, max_words)
    inputs = Input(name='inputs', shape=[max_len])
    layer = Embedding(max_words, 128, input_length=max_len)(inputs)
    layer = LSTM(256, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)(layer)
    layer = LSTM(256, dropout=0.5, recurrent_dropout=0.5)(layer)
    layer = Dense(Y_CLASS, name = 'out_layer')(layer)
    layer = Activation('softmax')(layer)
    model = Model(inputs=inputs, outputs=layer)
    return model

model = LSTM_Model(MAX_LEN, MAX_WORDS)
model.summary()
plot_model(model, to_file='./' + ODIR + '/model.png')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(  X_train,
            Y_train,
            batch_size=256,
            epochs=256,
            validation_split=0.25,
            callbacks = [EarlyStopping( monitor='val_loss',
				                        patience=5,
				                        min_delta=0.0001)])

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