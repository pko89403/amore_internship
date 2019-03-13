import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

training_data = './Input_json/train.json.csv'

df = pd.read_csv(training_data)
print(df.head(), df.info())

sns.countplot(df.cuisine)
plt.xlabel('Label')
plt.title('Number of cusine categories')
plt.show()

# Create input and output Vector
X = df.ingredients
Y = df.cuisine
Y_CLASS = 20
# Process the labels
le = LabelEncoder()
Y = le.fit_transform(Y)
Y = Y.reshape(-1, 1)
Y = to_categorical(Y, num_classes=Y_CLASS)
# Split into training and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15)

TOKEN_MAX_WORDS = 1000
TOKEN_MAX_LEN = 150
def tokenizeData(x_datas):

    """
    :param x_datas:
    Tokenize the data and convert the text to sequences.
    Add padding to ensure that all sequences have the same shape.

    :return: sequences_matrix
    """

    tok = Tokenizer(num_words=TOKEN_MAX_WORDS)
    tok.fit_on_texts(x_datas)
    sequences = tok.texts_to_sequences(x_datas)
    sequences_matrix = sequence.pad_sequences(sequences, maxlen=TOKEN_MAX_LEN)
    return sequences_matrix

ONEHOT_DIMENSION = 1000
ONEHOT_MAX_LEN = 10
def oneHotData(x_datas):

    """
    :param: x_datas:
    :return: results
    if word dict size is so big, using oneHot Hashing.
    data encoding using hashing,
    """
    results = np.zeros((len(x_datas), ONEHOT_MAX_LEN, ONEHOT_DIMENSION))
    for i, x_data in enumerate(x_datas):
        for j, word in list(enumerate(x_datas.split()))[:ONEHOT_MAX_LEN]:
            index = abs(hash(word)) % ONEHOT_DIMENSION
            results[i, j, index] = 1.

    return results
"""
WEM_FWORD = 10000
WEM_MAXLEN = 20
def wordEmbedding(x_datas):
"""
def RNN(max_len, max_words):
    inputs = Input(name='inputs', shape=[max_len])
    layer = Embedding(max_words, 50, input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256, name = 'FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(Y_CLASS, name = 'out_layer')(layer)
    layer = Activation('softmax')(layer)
    model = Model(inputs=inputs, outputs=layer)
    return model

model = RNN(TOKEN_MAX_LEN, TOKEN_MAX_WORDS)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(  tokenizeData(X_train),
            Y_train,
            batch_size=128,
            epochs=10,
            validation_split=0.2,
            callbacks = [EarlyStopping(monitor='val_loss', min_delta=0.0001)])



