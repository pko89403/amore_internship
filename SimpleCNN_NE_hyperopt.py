from __future__ import print_function

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from keras.preprocessing.text import *
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Reshape, Flatten, Concatenate, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from Data_proc import text2matrix, MAX_LEN, MAX_WORD, Y_CLASS, TRAINING_PATH


ODIR = 'CNN_NE_HypOpt'

def LSTM(X_train, Y_train, X_test, Y_test):
    inputs = Input(name='inputs', shape=(MAX_WORD,))
    reshape = Reshape((MAX_WORD, 1))(inputs)

    conv_0 = Conv1D(filters={{choice([64,128,256])}}, kernel_size={{choice([2,3,4])}}, padding='valid', activation='relu')(reshape)
    conv_1 = Conv1D(filters={{choice([64,128,256])}}, kernel_size={{choice([2,3,4])}}, padding='valid', activation='relu')(reshape)
    conv_2 = Conv1D(filters={{choice([64,128,256])}}, kernel_size={{choice([2,3,4])}}, padding='valid', activation='relu')(reshape)

    maxpool_0 = MaxPooling1D(pool_size={{choice([2,3])}}, padding='valid')(conv_0)
    maxpool_1 = MaxPooling1D(pool_size={{choice([2,3])}}, padding='valid')(conv_1)
    maxpool_2 = MaxPooling1D(pool_size={{choice([2,3])}}, padding='valid')(conv_2)

    flat_0 = Flatten()(maxpool_0)
    flat_1 = Flatten()(maxpool_1)
    flat_2 = Flatten()(maxpool_2)

    concatenated = Concatenate(axis=1)([flat_0, flat_1, flat_2])
    dropout = Dropout({{uniform(0, 1)}})(concatenated)

    output = Dense(units=Y_CLASS, activation='softmax')(dropout)
    model = Model(inputs=inputs, outputs=output)

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr={{choice([10**-3, 10**-2, 10**-1])}}), metrics=['accuracy'])

    model.fit(X_train,
              Y_train,
              batch_size={{choice([64,128])}},
              epochs=1024,
              shuffle=True,
              validation_split=0.2,
              callbacks=[EarlyStopping(monitor='val_loss',
                                       patience=16,
                                       )],
              verbose=2 )
    score = model.evaluate(X_test, Y_test)

    return {'loss': -score[1], 'status': STATUS_OK, 'model': model}


best_run, best_model = optim.minimize(model=LSTM,
                                      data=text2matrix,
                                      algo=tpe.suggest,
                                      max_evals=10,
                                      trials=Trials(),
                                      verbose=False)
best_model.summary()
X_train, X_test, Y_train, Y_test = text2matrix()
score = best_model.evaluate(X_test, Y_test)
print('Test loss: ', score[0])
print('Test Accuracy: ', score[1])
print(best_run)


import tensorflowjs as tfjs

tfjs.converters.save_keras_model(best_model, ODIR)
