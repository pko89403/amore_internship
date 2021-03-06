from __future__ import print_function

import pandas as pd
import keras
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Conv1D, MaxPool1D
from keras.layers import Reshape, Flatten, Dropout, Concatenate, BatchNormalization
from keras.models import Model
from keras.preprocessing.text import *
from keras.utils import plot_model
from keras.utils import to_categorical
from keras import backend as K
import tensorflowjs as tfjs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

training_data = '/content/drive/My Drive/colab_folder/Input_json/train.json.csv'
df = pd.read_csv(training_data)

# Create input and output Vector
X = df.ingredients
Y = df.cuisine
Y_CLASS = 20  # NUM OF CLASS

# Process the labels
le = LabelEncoder()
Y = le.fit_transform(Y)
Y = Y.reshape(-1, 1)
Y = to_categorical(Y, num_classes=Y_CLASS)

# Split into training and test data
MAX_WORDS = 0

def tokenizeData(x_datas):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x_datas)
    matrixes = tokenizer.texts_to_matrix(x_datas, mode='binary')
    vocab_size = len(tokenizer.word_index) + 1
    global MAX_WORDS
    MAX_WORDS = vocab_size
    return matrixes

X = tokenizeData(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

def CNN(max_words):
    inputs = Input(name='inputs', shape=(max_words,))
    Reshaped = Reshape(target_shape=(max_words, 1))(inputs)

    conv_0 = Conv1D(filters=64, kernel_size=3, padding='valid', activation='relu', kernel_regularizer='l2')(Reshaped)
    conv_01 = Conv1D(filters=64, kernel_size=3, padding='valid', activation='relu', kernel_regularizer='l2')(conv_0)
    conv_01_bn = BatchNormalization()(conv_01)
    maxpool_01 = MaxPool1D(pool_size=2, padding='valid')(conv_01_bn)
    conv_012 = Conv1D(filters=64, kernel_size=3, padding='valid', activation='relu', kernel_regularizer='l2')(maxpool_01)
    conv_0123 = Conv1D(filters=64, kernel_size=3, padding='valid', activation='relu', kernel_regularizer='l2')(conv_012)
    conv_0123_bn = BatchNormalization()(conv_0123)
    maxpool_012 = MaxPool1D(pool_size=2, padding='valid')(conv_0123_bn)
    dropout01 = Dropout(0.5)(maxpool_012)

    conv_1 = Conv1D(filters=64, kernel_size=4, padding='valid', activation='relu', kernel_regularizer='l2')(Reshaped)
    conv_11 = Conv1D(filters=64, kernel_size=4, padding='valid', activation='relu', kernel_regularizer='l2')(conv_1)
    conv_11_bn = BatchNormalization()(conv_11)
    maxpool_11 = MaxPool1D(pool_size=2, padding='valid')(conv_11_bn)
    conv_112 = Conv1D(filters=64, kernel_size=4, padding='valid', activation='relu', kernel_regularizer='l2')(maxpool_11)
    conv_1123 = Conv1D(filters=64, kernel_size=4, padding='valid', activation='relu', kernel_regularizer='l2')(conv_112)
    conv_1123_bn = BatchNormalization()(conv_1123)
    maxpool_112 = MaxPool1D(pool_size=2, padding='valid')(conv_1123_bn)
    dropout11 = Dropout(0.5)(maxpool_112)

    conv_2 = Conv1D(filters=64, kernel_size=5, padding='valid', activation='relu', kernel_regularizer='l2')(Reshaped)
    conv_21 = Conv1D(filters=64, kernel_size=5, padding='valid', activation='relu', kernel_regularizer='l2')(conv_2)
    conv_21_bn = BatchNormalization()(conv_21)
    maxpool_21 = MaxPool1D(pool_size=2, padding='valid')(conv_21_bn)
    conv_212 = Conv1D(filters=64, kernel_size=5, padding='valid', activation='relu', kernel_regularizer='l2')(maxpool_21)
    conv_2123 = Conv1D(filters=64, kernel_size=5, padding='valid', activation='relu', kernel_regularizer='l2')(conv_212)
    conv_2123_bn = BatchNormalization()(conv_2123)
    maxpool_212 = MaxPool1D(pool_size=2, padding='valid')(conv_2123_bn)
    dropout21 = Dropout(0.5)(maxpool_212)

    flat_0 = Flatten()(dropout01)
    flat_1 = Flatten()(dropout11)
    flat_2 = Flatten()(dropout21)

    concatenated = Concatenate(axis=1)([flat_0, flat_1, flat_2])

    dense1 = Dense(units=512, activation='relu', use_bias=False, kernel_regularizer='l2')(concatenated)
    dense1_bn = BatchNormalization()(dense1)
    dense1_drop = Dropout(0.5)(dense1_bn)
    dense2 = Dense(units=512, activation='relu', use_bias=False, kernel_regularizer='l2')(dense1_drop)
    dense2_bn = BatchNormalization()(dense2)
    dense2_drop = Dropout(0.5)(dense2_bn)
    dense3 = Dense(units=512, activation='relu', use_bias=False, kernel_regularizer='l2')(dense2_drop)
    dense3_bn = BatchNormalization()(dense3)
    dense3_drop = Dropout(0.5)(dense3_bn)

    global Y_CLASS
    output = Dense(units=Y_CLASS, activation='softmax')(dense3_drop)
    model = Model(inputs=inputs, outputs=output)

    return model

model = CNN(MAX_WORDS)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(X_train,
                    Y_train,
                    batch_size=200,
                    epochs=3,
                    validation_split=0.2,
                    callbacks=[EarlyStopping(monitor='val_loss',
                                             patience=5,
                                             )])

score = model.evaluate(X_test, Y_test)
print('Test loss: ', score[0])
print('Test Accuracy: ', score[1])

tfjs.converters.save_keras_model(model, '/content/drive/My Drive/colab_folder/')