from __future__ import print_function

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import *
from keras.preprocessing import sequence
from keras.utils import to_categorical

from Data_proc import text2seq, MAX_LEN, MAX_WORD, Y_CLASS, TRAINING_PATH
import numpy as np

from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Conv1D, MaxPool1D, Flatten
from keras.layers import BatchNormalization, Dropout, Concatenate
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

import tensorflowjs as tfjs

ODIR = 'MultiChannelCNN'
LOAD_WORD2VEC_WEIGHT = "./google300Weights.npy"
EMBEDDING_W = np.load(LOAD_WORD2VEC_WEIGHT)
EMBEDDING_DIM = 300

text2seq()


def MultiChannelCNN1(X_train, Y_train, X_test, Y_test):
    filterCnt = {{choice([128, 256, 512, 1024])}}

    EMBEDDING_W = np.load("./google300Weights.npy")
    inputs = Input(name='Inputs', shape=(MAX_LEN,))
    embedding = Embedding(input_dim=MAX_WORD,
                          output_dim=300,
                          weights=[EMBEDDING_W],
                          input_length=MAX_LEN,
                          trainable={{choice([True, False])}})(inputs)

    conv_0 = Conv1D(filters=filterCnt, kernel_size={{choice([2, 3, 4, 5, 6])}}, padding='valid',
                    kernel_regularizer='l2', activation='relu')(embedding)
    conv_0_bn = BatchNormalization()(conv_0)

    conv_1 = Conv1D(filters=filterCnt, kernel_size={{choice([2, 3, 4, 5, 6])}}, padding='valid',
                    kernel_regularizer='l2', activation='relu')(embedding)
    conv_1_bn = BatchNormalization()(conv_1)

    conv_2 = Conv1D(filters=filterCnt, kernel_size={{choice([2, 3, 4, 5, 6])}}, padding='valid',
                    kernel_regularizer='l2', activation='relu')(embedding)
    conv_2_bn = BatchNormalization()(conv_2)

    maxpool_0 = MaxPool1D(pool_size=2, padding='valid')(conv_0_bn)
    maxpool_1 = MaxPool1D(pool_size=2, padding='valid')(conv_1_bn)
    maxpool_2 = MaxPool1D(pool_size=2, padding='valid')(conv_2_bn)

    flat_0 = Flatten()(maxpool_0)
    flat_1 = Flatten()(maxpool_1)
    flat_2 = Flatten()(maxpool_2)

    embedding2 = Embedding(input_dim=MAX_WORD,
                           output_dim=300,
                           input_length=MAX_LEN)(inputs)

    conv_20 = Conv1D(filters=filterCnt, kernel_size={{choice([2, 3, 4, 5, 6])}}, padding='valid',
                     kernel_regularizer='l2', activation='relu')(embedding2)
    conv_20_bn = BatchNormalization()(conv_20)

    conv_21 = Conv1D(filters=filterCnt, kernel_size={{choice([2, 3, 4, 5, 6])}}, padding='valid',
                     kernel_regularizer='l2', activation='relu')(embedding2)
    conv_21_bn = BatchNormalization()(conv_21)

    conv_22 = Conv1D(filters=filterCnt, kernel_size={{choice([2, 3, 4, 5, 6])}}, padding='valid',
                     kernel_regularizer='l2', activation='relu')(embedding2)
    conv_22_bn = BatchNormalization()(conv_22)

    maxpool_20 = MaxPool1D(pool_size=2, padding='valid')(conv_20_bn)
    maxpool_21 = MaxPool1D(pool_size=2, padding='valid')(conv_21_bn)
    maxpool_22 = MaxPool1D(pool_size=2, padding='valid')(conv_22_bn)

    flat_20 = Flatten()(maxpool_20)
    flat_21 = Flatten()(maxpool_21)
    flat_22 = Flatten()(maxpool_22)

    concatenated0 = Concatenate(axis=1)([flat_0, flat_1, flat_2])
    dropout0 = Dropout({{uniform(0, 1)}})(concatenated0)

    concatenated1 = Concatenate(axis=1)([flat_20, flat_21, flat_22])
    dropout1 = Dropout({{uniform(0, 1)}})(concatenated1)

    concatenatedA = Concatenate(axis=1)([dropout0, dropout1])
    dropoutA = Dropout({{uniform(0, 1)}})(concatenatedA)

    output = Dense(units=Y_CLASS, activation='softmax')(dropoutA)

    model = Model(inputs=inputs, outputs=output)

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr={{choice([10 ** -3, 10 ** -2, 10 ** -1])}}),
                  metrics=['accuracy'])
    model.fit(X_train, Y_train,
              batch_size={{choice([128, 256, 512])}},
              epochs=1024,
              validation_split=0.2,
              shuffle=True,
              callbacks=[EarlyStopping(monitor='val_loss', patience=16, )],
              verbose=2)

    score = model.evaluate(X_test, Y_test)
    return {'loss': -score[1], 'status': STATUS_OK, 'model': model}


best_run, best_model = optim.minimize(model=MultiChannelCNN1,
                                      data=text2seq,
                                      algo=tpe.suggest,
                                      max_evals=10,
                                      trials=Trials(),
                                      verbose=False)

best_model.summary()
plot_model(best_model, to_file='./' + ODIR + '/model.png')
X_train, X_test, Y_train, Y_test = text2seq()
score = best_model.evaluate(X_test, Y_test)
print('Test loss: ', score[0])
print('Test Accuracy: ', score[1])
print(best_run)
tfjs.converters.save_keras_model(best_model, ODIR)
