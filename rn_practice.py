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

from keras.models import Model
import keras.backend as K
from keras.layers import Input, Embedding, Lambda
inputs = Input(shape=(x_limitLen,))
embedding = Embedding(input_dim = x_maxWords,
                      output_dim= 300,
                      input_length=x_limitLen)(inputs)
encoder = Lambda(lambda x : K.sum(x, axis=1),
                 output_shape=lambda shape: (shape[0],) + shape[2:])(embedding)
