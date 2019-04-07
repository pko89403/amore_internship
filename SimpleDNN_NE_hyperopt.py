from __future__ import print_function

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import *
from keras.utils import to_categorical

from Data_proc import text2matrix, MAX_LEN, MAX_WORD, Y_CLASS, TRAINING_PATH

from keras.models import Model
from keras.layers import Input, Dense
from keras.layers import Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform


ODIR = 'DNN_NE_HypOpt'

def DNN(X_train, Y_train, X_test, Y_test):
    inputs = Input(name='inputs', shape=(MAX_WORD,))

    dense1 = Dense(units={{choice([256,512,1024,2048])}}, activation='relu')(inputs)
    dense1_BN = BatchNormalization()(dense1)
    dropOut1 = Dropout({{uniform(0, 1)}})(dense1_BN)

    dense2 = Dense(units={{choice([256,512,1024,2048])}}, activation='relu')(dropOut1)
    dense2_BN = BatchNormalization()(dense2)
    dropOut2 = Dropout({{uniform(0, 1)}})(dense2_BN)

    dense3 = Dense(units={{choice([256,512,1024,2048])}}, activation='relu')(dropOut2)
    dense3_BN = BatchNormalization()(dense3)
    dropOut3 = Dropout({{uniform(0, 1)}})(dense3_BN)

    dense4 = Dense(units={{choice([256,512,1024,2048])}}, activation='relu')(dropOut3)
    dense4_BN = BatchNormalization()(dense4)
    dropOut4 = Dropout({{uniform(0, 1)}})(dense4_BN)

    output = Dense(units=Y_CLASS, activation='softmax')(dropOut4)
    model = Model(inputs=inputs, outputs=output)

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr={{choice([10**-3, 10**-2, 10**-1])}}), metrics=['accuracy'])

    model.fit(X_train,
              Y_train,
              batch_size={{choice([128, 256])}},
              epochs=1024,
              validation_split=0.2,
              callbacks=[EarlyStopping(monitor='val_loss',
                                       patience=8,
                                       )],
              verbose=2 )

    score = model.evaluate(X_test, Y_test)

    return {'loss': -score[1], 'status': STATUS_OK, 'model': model}


best_run, best_model = optim.minimize(model=DNN,
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