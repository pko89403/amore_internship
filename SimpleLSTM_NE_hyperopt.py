from __future__ import print_function

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from keras.preprocessing.text import *
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Reshape
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform


ODIR = 'LSTM_NE_HypOpt'
MAX_WORDS = 0

def data():
    training_data = './Input_json/train.json.csv'
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
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X)
    matrixes = tokenizer.texts_to_matrix(X, mode='binary')

    vocab_size = len(tokenizer.word_index) + 1

    global MAX_WORDS
    MAX_WORDS = vocab_size

    X = matrixes
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
    return X_train, X_test, Y_train, Y_test

def LSTM(X_train, Y_train, X_test, Y_test):
    global MAX_WORDS
    inputs = Input(name='inputs', shape=(MAX_WORDS,))
    reshape = Reshape((1, MAX_WORDS))(inputs)

    layer = LSTM(units={{choice([128, 256, 512, 1024])}}, dropout={{uniform(0, 1)}},
                 recurrent_dropout={{uniform(0, 1)}}, return_sequences=True)(reshape)
    layer = LSTM(units={{choice([128, 256, 512, 1024])}}, dropout={{uniform(0, 1)}},
                 recurrent_dropout={{uniform(0, 1)}}, return_sequences=True)(layer)
    layer = LSTM(units={{choice([128, 256, 512, 1024])}}, dropout={{uniform(0, 1)}},
                 recurrent_dropout={{uniform(0, 1)}}, return_sequences=True)(layer)
    layer = LSTM(units={{choice([128, 256, 512, 1024])}}, dropout={{uniform(0, 1)}},
                 recurrent_dropout={{uniform(0, 1)}}, return_sequences=True)(layer)
    layer = LSTM(units={{choice([128, 256, 512, 1024])}}, dropout={{uniform(0, 1)}},
                 recurrent_dropout={{uniform(0, 1)}})(layer)

    global Y_CLASS
    output = Dense(units=Y_CLASS, activation='softmax')(layer)
    model = Model(inputs=inputs, outputs=output)

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr={{choice([10**-3, 10**-2, 10**-1])}}), metrics=['accuracy'])

    model.fit(X_train,
              Y_train,
              batch_size={{choice([128, 256, 512])}},
              epochs=1024,
              validation_split=0.2,
              callbacks=[EarlyStopping(monitor='val_loss',
                                       patience=16,
                                       )],
              verbose=2 )

    score = model.evaluate(X_test, Y_test)

    return {'loss': -score[1], 'status': STATUS_OK, 'model': model}


best_run, best_model = optim.minimize(model=LSTM,
                                      data=data,
                                      algo=tpe.suggest,
                                      max_evals=10,
                                      trials=Trials(),
                                      verbose=False)
best_model.summary()
X_train, X_test, Y_train, Y_test = data()
score = best_model.evaluate(X_test, Y_test)
print('Test loss: ', score[0])
print('Test Accuracy: ', score[1])
print(best_run)


import tensorflowjs as tfjs

tfjs.converters.save_keras_model(best_model, ODIR)