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
from keras.layers import Input, Dense, Embedding, Lambda, LSTM, Reshape
import keras.backend as K
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

import tensorflowjs as tfjs


ODIR = 'Transfer_CBOW_LSTM'

def CBOW_LSTM(X_train, Y_train, X_test, Y_test):
    EMBEDDING_W = np.load("./google300Weights.npy")

    inputs = Input(shape=(MAX_LEN,))
    embedding = Embedding(input_dim = MAX_WORD,
                          output_dim= 300,
                          weights = [EMBEDDING_W],
                          input_length=MAX_LEN,
                          trainable=False)(inputs)

    encoder=Lambda(lambda x : K.mean(x, axis=1),
                    output_shape=lambda shape: (shape[0], 1) + shape[2:])( embedding )

    reshape = Reshape((1, 300))(encoder)

    layer = LSTM({{choice([128,256,512,1024])}}, dropout={{uniform(0, 1)}}, recurrent_dropout={{uniform(0, 1)}}, return_sequences=True)(reshape)
    layer = LSTM(units={{choice([256,512,1024])}}, dropout={{uniform(0, 1)}}, recurrent_dropout={{uniform(0, 1)}}, return_sequences=True)(layer)
    layer = LSTM(units={{choice([256,512,1024])}}, dropout={{uniform(0, 1)}}, recurrent_dropout={{uniform(0, 1)}}, return_sequences=True)(layer)
    layer = LSTM(units={{choice([256,512,1024])}}, dropout={{uniform(0, 1)}}, recurrent_dropout={{uniform(0, 1)}}, return_sequences=True)(layer)
    layer = LSTM(units={{choice([256,512,1024])}}, dropout={{uniform(0, 1)}}, recurrent_dropout={{uniform(0, 1)}})(layer)

    output = Dense(units=Y_CLASS, activation = 'softmax')(layer)

    model = Model(inputs = inputs, output=output)

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

best_run, best_model = optim.minimize(model=CBOW_LSTM,
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