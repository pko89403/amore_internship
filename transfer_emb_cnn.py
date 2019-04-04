import pandas as pd

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


ODIR = 'TRANSFER_EMB_CNN'
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

x_limitLen = 30
sequence_matrix = sequence.pad_sequences(sequences, maxlen= x_limitLen)

X = sequence_matrix
Y = transform_reshape_categorical_Y

X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.25,
                                                    )

import numpy as np
embedding_weights = np.load(LOAD_WORD2VEC_WEIGHT)
EMBEDDING_DIM = 300
vocabulary_size = x_maxWords

from keras.models import Model
from keras.layers import Input, Reshape, Embedding, Dense, Conv2D, MaxPool2D
from keras.layers import Flatten, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping

inputs = Input(name='Inputs', shape=(x_limitLen,))
embedding= Embedding(   input_dim = vocabulary_size,
                        output_dim = EMBEDDING_DIM,
                        weights = [embedding_weights],
                        input_length = x_limitLen,
                        trainable=False)(inputs)

reshape = Reshape((x_limitLen, EMBEDDING_DIM, 1))(embedding)

conv_0 = Conv2D(filters=128, kernel_size=(1,1), padding='valid', kernel_regularizer='l2', activation='relu')(reshape)
conv_0_bn = BatchNormalization()(conv_0)
maxpool_0 = MaxPool2D(pool_size=(2,2), padding='valid')(conv_0_bn)
dropout_0 = Dropout(0.5)(maxpool_0)


conv_1 = Conv2D(filters=128, kernel_size=(1,1), padding='valid',  kernel_regularizer='l2', activation='relu')(dropout_0)
conv_1_bn = BatchNormalization()(conv_1)
maxpool_1 = MaxPool2D(pool_size=(2,2), padding='valid')(conv_1_bn)
dropout_1 = Dropout(0.5)(maxpool_1)

conv_2 = Conv2D(filters=128, kernel_size=(1,1), padding='valid',  kernel_regularizer='l2', activation='relu')(dropout_1)
conv_2_bn = BatchNormalization()(conv_2)
maxpool_2 = MaxPool2D(pool_size=(2,2), padding='valid')(conv_2_bn)
dropout_2 = Dropout(0.5)(maxpool_2)

flat_0 = Flatten()(dropout_2)

output = Dense(units=Y_CLASS, activation='softmax')(flat_0)

model = Model(inputs=inputs, outputs=output)

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train,
                    Y_train,
                    batch_size=200,
                    epochs=256,
                    validation_split=0.2,
                    callbacks = [   EarlyStopping(	monitor='val_loss',
							                        patience=5)])

score = model.evaluate(X_test, Y_test)
print('Test loss: ', score[0])
print('Test Accuracy: ', score[1])

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('./' + ODIR + '/train_acc.png')
plt.clf()
plt.cla()
plt.close()
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('./' + ODIR + '/train_loss.png')
plt.clf()
plt.cla()
plt.close()



import tensorflowjs as tfjs
tfjs.converters.save_keras_model(model, ODIR)