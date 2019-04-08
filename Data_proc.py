import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import *
from keras.preprocessing import sequence
from keras.utils import to_categorical

MAX_WORD=3065
MAX_LEN=30
Y_CLASS=20
TRAINING_PATH = './Input_json/train.json.csv'

def text2seq():
    MAX_LEN = 30
    Y_CLASS = 20
    TRAINING_PATH = './Input_json/train.json.csv'

    df = pd.read_csv(TRAINING_PATH)

    # Create input and output Vector
    X = df.ingredients
    Y = df.cuisine
    # Process the labels
    le = LabelEncoder()
    Y = le.fit_transform(Y)
    Y = Y.reshape(-1,1)
    Y = to_categorical(Y, num_classes=Y_CLASS)

    # Split into training and test data

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X)

    X = sequence.pad_sequences(sequences, maxlen=MAX_LEN)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=None, shuffle=False)
    return X_train, X_test, Y_train, Y_test

def text2matrix():
    Y_CLASS = 20
    TRAINING_PATH = './Input_json/train.json.csv'

    df = pd.read_csv(TRAINING_PATH)
    # Create input and output Vector
    X = df.ingredients
    Y = df.cuisine

    # Process the labels
    le = LabelEncoder()
    Y = le.fit_transform(Y)
    Y = Y.reshape(-1, 1)
    Y = to_categorical(Y, num_classes=Y_CLASS)
    # Split into training and test data
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X)
    matrixes = tokenizer.texts_to_matrix(X, mode='binary')

    X = matrixes
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=None, shuffle=False)
    return X_train, X_test, Y_train, Y_test
