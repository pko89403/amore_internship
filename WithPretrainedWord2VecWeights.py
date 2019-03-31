import pandas as pd

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split


ODIR = 'PretrainedWithCNN'
TRAINING_DATA_PATH = './Input_json/train.json.csv'

Y_CLASS = 20
LOAD_WORD2VEC_PATH = '../pretrained_emb_weights/GoogleNews-vectors-negative300.bin.gz'



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

import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.models.keyedvectors import KeyedVectors

word_vectors = KeyedVectors.load_word2vec_format(LOAD_WORD2VEC_PATH,
                                                 binary=True)

import numpy as np

EMBEDDING_DIM = 300
vocabulary_size = x_maxWords

embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))
for word, i in tokenizer.word_index.items():
    if( i >= vocabulary_size):
        continue
    try:
        embedding_vector = word_vectors[word]
        embedding_matrix[i] = embedding_vector
    except KeyError:
        embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25), EMBEDDING_DIM)

del(word_vectors)

from keras.models import Model
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate, BatchNormalization
from keras.callbacks import EarlyStopping

inputs = Input(name='Inputs', shape=(x_limitLen,))
embedding= Embedding(input_dim = vocabulary_size,
                            output_dim = EMBEDDING_DIM,
                            weights = [embedding_matrix],
                            input_length = x_limitLen,
                            trainable=True)
reshape = Reshape((x_limitLen, 300, 1))(embedding)

conv_0 = Conv2D(filters=256, kernel_size=(3, 64), padding='valid', kernel_regularizer='l2', activation='relu')(reshape)
conv_0_bn = BatchNormalization()(conv_0)

conv_1 = Conv2D(filters=256, kernel_size=(4, 64), padding='valid',  kernel_regularizer='l2', activation='relu')(reshape)
conv_1_bn = BatchNormalization()(conv_1)

conv_2 = Conv2D(filters=256, kernel_size=(5, 64), padding='valid',  kernel_regularizer='l2', activation='relu')(reshape)
conv_2_bn = BatchNormalization()(conv_2)

maxpool_0 = MaxPool2D(pool_size=(x_limitLen - 3 + 1, 1), strides=(1,1), padding='valid')(conv_0_bn)
maxpool_1 = MaxPool2D(pool_size=(x_limitLen - 4 + 1, 1), strides=(1,1), padding='valid')(conv_1_bn)
maxpool_2 = MaxPool2D(pool_size=(x_limitLen - 5 + 1, 1), strides=(1,1), padding='valid')(conv_2_bn)

concatenated = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
flattened = Flatten()(concatenated)
dropout = Dropout(0.5)(flattened)

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
							        patience=5,
							        min_delta=0.0001)])

score = model.evaluate(X_test, Y_test)
print('Test loss: ', score[0])
print('Test Accuracy: ', score[1])

import tensorflowjs as tfjs
tfjs.converters.save_keras_model(model, ODIR)