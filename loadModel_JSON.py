import tensorflowjs as tfjs
import pandas as pd

from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import *
from keras.utils import to_categorical
from keras.models import Model

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
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    return X_train, X_test, Y_train, Y_test


loadModels = tfjs.converters.load_keras_model('./DNN_NE_HypOpt/model.json')
loadModels.save('my_hyp.h5')

model = load_model('my_hyp.h5')
model.summary()
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

X_train, X_test, Y_train, Y_test = data()
score = model.evaluate(X_test, Y_test)
print(score[0], score[1])