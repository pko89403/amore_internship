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
from keras.layers import Input, Dense, Embedding, Lambda, Reshape, Conv1D, MaxPooling1D, Flatten, Concatenate, Dropout
import keras.backend as K
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

import tensorflowjs as tfjs


ODIR = 'Transfer_CBOW_CNN'

def CBOW_CNN(X_train, Y_train, X_test, Y_test):
    EMBEDDING_W = np.load("./google300Weights.npy")

    inputs = Input(shape=(MAX_LEN,))
    embedding = Embedding(input_dim = MAX_WORD,
                          output_dim= 300,
                          weights = [EMBEDDING_W],
                          input_length=MAX_LEN,
                          trainable={{choice([True, False])}})(inputs)

    encoder=Lambda(lambda x : K.mean(x, axis=1),
                    output_shape=lambda shape: (shape[0], 1) + shape[2:])( embedding )

    reshape = Reshape((1, 300))(encoder)

    conv_0 = Conv1D(filters={{choice([128,256,512,1024])}}, kernel_size={{choice([2,3,4])}}, padding='valid', activation='relu')(embedding)
    conv_1 = Conv1D(filters={{choice([128,256,512,1024])}}, kernel_size={{choice([2,3,4])}}, padding='valid', activation='relu')(embedding)
    conv_2 = Conv1D(filters={{choice([128,256,512,1024])}}, kernel_size={{choice([2,3,4])}}, padding='valid', activation='relu')(embedding)

    maxpool_0 = MaxPooling1D(pool_size={{choice([2,3])}}, padding='valid')(conv_0)
    maxpool_1 = MaxPooling1D(pool_size={{choice([2,3])}}, padding='valid')(conv_1)
    maxpool_2 = MaxPooling1D(pool_size={{choice([2,3])}}, padding='valid')(conv_2)

    flat_0 = Flatten()(maxpool_0)
    flat_1 = Flatten()(maxpool_1)
    flat_2 = Flatten()(maxpool_2)

    concatenated = Concatenate(axis=1)([flat_0, flat_1, flat_2])
    dropout = Dropout({{uniform(0, 1)}})(concatenated)

    output = Dense(units=Y_CLASS, activation = 'softmax')(dropout)

    model = Model(inputs = inputs, output=output)

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr={{choice([10 ** -3, 10 ** -2, 10 ** -1])}}),
                  metrics=['accuracy'])

    model.fit(X_train,
              Y_train,
              batch_size={{choice([128, 256, 512])}},
              epochs=1024,
              shuffle=True,
              validation_split=0.2,
              callbacks=[EarlyStopping(monitor='val_loss',patience=16,)],
              verbose=2)

    score = model.evaluate(X_test, Y_test)
    return {'loss': -score[1], 'status': STATUS_OK, 'model': model}

best_run, best_model = optim.minimize(model=CBOW_CNN,
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