import pandas as pd

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split


ODIR = 'PretrainedWithCNN'
TRAINING_DATA_PATH = './Input_json/train.json.csv'

Y_CLASS = 20
LOAD_WORD2VEC_WEIGHT = "./google300Weights.npy"



df = pd.read_csv(TRAINING_DATA_PATH)

X = df.ingredients
Y = df.cuisine

le = LabelEncoder()
transform_Y = le.fit_transform(Y)
transform_reshape_Y = transform_Y.reshape(-1, 1)
transform_reshape_categorical_Y = to_categorical(transform_reshape_Y,
                                                 num_classes=Y_CLASS)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)

x_maxLen = max([len(x) - 1 for x in sequences])
x_maxWords = len(tokenizer.word_index) + 1

print('Tokenizing Result', x_maxLen, ', ', x_maxWords)
x_limitLen = 30
sequence_matrix = sequence.pad_sequences(sequences, maxlen= x_limitLen)

X = sequence_matrix
Y = transform_reshape_categorical_Y

X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.25,
                                                    random_state=42)

import numpy as np
embedding_weights = np.load(LOAD_WORD2VEC_WEIGHT)
EMBEDDING_DIM = 300


from keras.models import Model
import keras.backend as K
from keras.layers import Input, Embedding, Lambda, Dense, Dropout
inputs = Input(shape=(x_limitLen,))
embedding = Embedding(input_dim = x_maxWords,
                      output_dim= 300,
                      weights = [embedding_weights],
                      input_length=x_limitLen,
                      trainable=False)(inputs)
encoder = Lambda(lambda x : K.mean(x, axis=1),
                 output_shape=lambda shape: (shape[0], ) + shape[2:])( embedding )

dense0 = Dense(units=1024, activation = 'relu')(encoder)
dropout0 = Dropout(0.5)(dense0)
dense1 = Dense(units=1024, activation = 'relu')(dense0)
dropout1 = Dropout(0.5)(dense1)
dense2 = Dense(units=1024, activation = 'relu')(dense1)
dropout2 = Dropout(0.5)(dense2)
dense3 = Dense(units=1024, activation = 'relu')(dense2)
dropout3 = Dropout(0.5)(dense3)


output = Dense(units=Y_CLASS, activation = 'softmax')(dense3)



model = Model(inputs = inputs, output=output)

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from keras.callbacks import EarlyStopping

history = model.fit(X_train,
                    Y_train,
                    batch_size=200,
                    epochs=256,
                    validation_split=0.2,
                    callbacks = [EarlyStopping(monitor='val_loss',
                                               patience=10,
                                               min_delta=0.001)]
                    )

score = model.evaluate(X_test, Y_test)
print('Test loss: ', score[0])
print('Test Accuracy: ', score[1])

