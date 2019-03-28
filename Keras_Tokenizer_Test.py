from __future__ import print_function

import pandas as pd
from sklearn.preprocessing import LabelEncoder

import io
import json

from keras.preprocessing.text import *
from keras.preprocessing import sequence
from keras.utils import to_categorical


training_data = './Input_json/train.json.csv'

df = pd.read_csv(training_data)

# Create input and output Vector
X = df.ingredients
Y = df.cuisine

# Split into training and test data
MAX_WORDS = 0
MAX_LEN = 0

def tokenizeData(x_datas):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x_datas)
    sequences = tokenizer.texts_to_sequences(x_datas)
    print(x_datas[0], sequences[0])
    
    maxlen = max([len(x) - 1 for x in sequences])
    vocab_size = len(tokenizer.word_index)+1

    print(tokenizer.texts_to_sequences('romaine lettuce black olives grape tomatoes garlic pepper purple onion seasoning garbanzo beans feta cheese crumbles'))   
    global MAX_WORDS
    MAX_WORDS = vocab_size
    global MAX_LEN
    MAX_LEN = 150
    sequences_matrix = sequence.pad_sequences(sequences, maxlen=MAX_LEN)
    print(sequences_matrix)
    tok_json = tokenizer.to_json()
    with io.open('tokenizer2.json', 'w', encoding='utf-8') as f:
       f.write(json.dumps(tok_json, ensure_ascii=False))
    return sequences_matrix

X = tokenizeData(X)

