import pandas as pd

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split


ODIR = 'MultiChannelCNN'
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
matrixes = tokenizer.texts_to_matrix(X, mode='binary')

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
                                                    random_state=50000)

import numpy as np
embedding_weights = np.load(LOAD_WORD2VEC_WEIGHT)
EMBEDDING_DIM = 300
vocabulary_size = x_maxWords

from keras.models import Model
from keras.layers import Input, Embedding, Dense, Conv1D, MaxPool1D
from keras.layers import Flatten, Dropout, Concatenate, BatchNormalization
from keras.callbacks import EarlyStopping

inputs = Input(name='Inputs', shape=(x_limitLen,))
embedding= Embedding(   input_dim = vocabulary_size,
                        output_dim = EMBEDDING_DIM,
                        weights = [embedding_weights],
                        input_length = x_limitLen,
                        trainable=True)(inputs)

conv_0 = Conv1D(filters=256, kernel_size=3, padding='valid', kernel_regularizer='l2', activation='relu')(embedding)
conv_0_bn = BatchNormalization()(conv_0)
conv_0_bn = Dropout(0.5)(conv_0_bn)

conv_1 = Conv1D(filters=256, kernel_size=4, padding='valid',  kernel_regularizer='l2', activation='relu')(embedding)
conv_1_bn = BatchNormalization()(conv_1)
conv_1_bn = Dropout(0.5)(conv_1_bn)

conv_2 = Conv1D(filters=256, kernel_size=5, padding='valid',  kernel_regularizer='l2', activation='relu')(embedding)
conv_2_bn = BatchNormalization()(conv_2)
conv_2_bn = Dropout(0.5)(conv_2_bn)

maxpool_0 = MaxPool1D(pool_size=2, padding='valid')(conv_0_bn)
maxpool_1 = MaxPool1D(pool_size=2, padding='valid')(conv_1_bn)
maxpool_2 = MaxPool1D(pool_size=2, padding='valid')(conv_2_bn)

flat_0 = Flatten()(maxpool_0)
flat_1 = Flatten()(maxpool_1)
flat_2 = Flatten()(maxpool_2)

embedding2 = Embedding(input_dim=vocabulary_size, output_dim=EMBEDDING_DIM, input_length=x_limitLen)(inputs)

conv_20 = Conv1D(filters=256, kernel_size=3, padding='valid', kernel_regularizer='l2', activation='relu')(embedding2)
conv_20_bn = BatchNormalization()(conv_20)
conv_20_bn = Dropout(0.5)(conv_20_bn)

conv_21 = Conv1D(filters=256, kernel_size=4, padding='valid',  kernel_regularizer='l2', activation='relu')(embedding2)
conv_21_bn = BatchNormalization()(conv_21)
conv_21_bn = Dropout(0.5)(conv_21_bn)

conv_22 = Conv1D(filters=256, kernel_size=5, padding='valid',  kernel_regularizer='l2', activation='relu')(embedding2)
conv_22_bn = BatchNormalization()(conv_22)
conv_22_bn = Dropout(0.5)(conv_22_bn)


maxpool_20 = MaxPool1D(pool_size=2, padding='valid')(conv_20_bn)
maxpool_21 = MaxPool1D(pool_size=2, padding='valid')(conv_21_bn)
maxpool_22 = MaxPool1D(pool_size=2, padding='valid')(conv_22_bn)

flat_20 = Flatten()(maxpool_20)
flat_21 = Flatten()(maxpool_21)
flat_22 = Flatten()(maxpool_22)


concatenated = Concatenate(axis=1)([flat_0, flat_1, flat_2, flat_20, flat_21, flat_22])
dropout = Dropout(0.6)(concatenated)

output = Dense(units=Y_CLASS, activation='softmax')(dropout)

model = Model(inputs=inputs, outputs=output)

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train,
                    Y_train,
                    batch_size=200,
                    epochs=256,
                    validation_split=0.2,
                    callbacks = [   EarlyStopping(	monitor='val_loss',
							        patience=10,
							        min_delta=0.001)])

score = model.evaluate(X_test, Y_test)
print('Test loss: ', score[0])
print('Test Accuracy: ', score[1])

import tensorflowjs as tfjs
tfjs.converters.save_keras_model(model, ODIR)