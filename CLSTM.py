from __future__ import print_function

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import *
from keras.preprocessing import sequence
from keras.utils import to_categorical

from Data_proc import text2seq, MAX_LEN, MAX_WORD, Y_CLASS, TRAINING_PATH

from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Conv1D, MaxPool1D, LSTM
from keras.layers import Dropout, Concatenate
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

import tensorflowjs as tfjs

ODIR = 'CNN_LSTM_Model'


def CNN_LSTM(X_train, Y_train, X_test, Y_test):
    filterCnt = {{choice([128,256,512,1024])}}

    inputs = Input(name='inputs', shape=(MAX_LEN,))
    embedding = Embedding(input_dim=MAX_WORD, output_dim=300, input_length=MAX_LEN)(inputs)

    conv_0 = Conv1D(filters=filterCnt, kernel_size={{choice([2,3,4,5,6])}}, padding='valid', kernel_initializer='normal', activation='relu')(embedding)
    conv_1 = Conv1D(filters=filterCnt, kernel_size={{choice([2,3,4,5,6])}}, padding='valid', kernel_initializer='normal', activation='relu')(embedding)
    conv_2 = Conv1D(filters=filterCnt, kernel_size={{choice([2,3,4,5,6])}}, padding='valid', kernel_initializer='normal', activation='relu')(embedding)

    maxpool_0 = MaxPool1D(pool_size={{choice([2,3])}}, padding='valid')(conv_0)
    maxpool_1 = MaxPool1D(pool_size={{choice([2,3])}}, padding='valid')(conv_1)
    maxpool_2 = MaxPool1D(pool_size={{choice([2,3])}}, padding='valid')(conv_2)

    concatenated = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])

    lstm_0 = LSTM({{choice([128,256,512,1024])}}, dropout={{uniform(0, 1)}}, recurrent_dropout={{uniform(0, 1)}})(concatenated)
    dropout = Dropout({{uniform(0, 1)}})(lstm_0)

    output = Dense(units=Y_CLASS, activation='softmax')(dropout)
    model = Model(inputs=inputs, outputs=output)

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr={{choice([10 ** -3, 10 ** -2, 10 ** -1])}}),
                  metrics=['accuracy'])

    model.fit(X_train,
              Y_train,
              batch_size={{choice([128, 256])}},
              epochs=1024,
              validation_split=0.2,
              callbacks=[EarlyStopping(monitor='val_loss',patience=8,)],
              verbose=2)

    score = model.evaluate(X_test, Y_test)
    return {'loss': -score[1], 'status': STATUS_OK, 'model': model}

best_run, best_model = optim.minimize(model=CNN_LSTM,
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