from __future__ import print_function

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Conv1D, MaxPool1D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.preprocessing.text import *
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import tensorflowjs as tfjs

ODIR = '1D_YoonKim_Model'
training_data = './Input_json/train.json.csv'

df = pd.read_csv(training_data)

# Create input and output Vector
X = df.ingredients
Y = df.cuisine
Y_CLASS = 20 # NUM OF CLASS
# Process the labels

le = LabelEncoder()
Y = le.fit_transform(Y)
Y = Y.reshape(-1,1)
Y = to_categorical(Y, num_classes=Y_CLASS)

# Split into training and test data
MAX_WORDS = 0
MAX_LEN = 0

def tokenizeData(x_datas):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(x_datas)
	sequences = tokenizer.texts_to_sequences(x_datas)
    
	vocab_size = len(tokenizer.word_index)+1

	global MAX_WORDS, MAX_LEN
	MAX_WORDS = vocab_size
	MAX_LEN = 30
	sequences_matrix = sequence.pad_sequences(sequences, maxlen=MAX_LEN)

	return sequences_matrix

X = tokenizeData(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

def CNN(max_len, max_words):
	print(max_len, max_words)
	inputs = Input(name='inputs', shape=(max_len,))
	embedding = Embedding(input_dim=max_words, output_dim=30, input_length=max_len)(inputs)

	conv_0 = Conv1D(filters=512, kernel_size=3, padding='valid', kernel_initializer='normal', activation='relu', kernel_regularizer='l2')(embedding)
	conv_1 = Conv1D(filters=512, kernel_size=4, padding='valid', kernel_initializer='normal', activation='relu', kernel_regularizer='l2')(embedding)
	conv_2 = Conv1D(filters=512, kernel_size=5, padding='valid', kernel_initializer='normal', activation='relu', kernel_regularizer='l2')(embedding)
	  
	maxpool_0 = MaxPool1D(pool_size=2, padding='valid')(conv_0)
	maxpool_1 = MaxPool1D(pool_size=2, padding='valid')(conv_1)
	maxpool_2 = MaxPool1D(pool_size=2, padding='valid')(conv_2)
	
	flat_0 = Flatten()(maxpool_0)
	flat_1 = Flatten()(maxpool_1)
	flat_2 = Flatten()(maxpool_2)

	concatenated = Concatenate(axis=1)([flat_0, flat_1, flat_2])
	dropout = Dropout(0.5)(concatenated)
	
	global Y_CLASS
	output = Dense(units=Y_CLASS, activation='softmax')(dropout)
	
	model = Model(inputs=inputs, outputs=output)
	
	return model

model = CNN(MAX_LEN, MAX_WORDS)
model.summary()
plot_model(model, to_file='./'+ ODIR + '/model.png')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train,
					Y_train,
					batch_size=200,
					epochs=256,
					validation_split=0.2,
					callbacks = [EarlyStopping(	monitor='val_loss',patience=5)])

# Plot training & validation accuracy values
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

score = model.evaluate(X_test, Y_test)
print('Test loss: ', score[0])
print('Test Accuracy: ', score[1])

tfjs.converters.save_keras_model(model, ODIR)