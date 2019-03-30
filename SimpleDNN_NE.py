from __future__ import print_function

import pandas as pd
import io
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from keras.preprocessing.text import *
from keras.utils import plot_model, to_categorical
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers import Dropout, BatchNormalization
from keras.callbacks import EarlyStopping

import tensorflowjs as tfjs

ODIR = 'DNN_NE'
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

def tokenizeData(x_datas):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(x_datas)
	matrixes = tokenizer.texts_to_matrix(x_datas, mode='binary')

	vocab_size = len(tokenizer.word_index) + 1
	print(' Bag Of words : ', vocab_size)

	global MAX_WORDS
	MAX_WORDS = vocab_size

	print(' Examples : ', len(matrixes), len(matrixes[0]) )
	print(x_datas[0] + ' -> ' + matrixes[0])
	tok_json = tokenizer.to_json()
	with io.open('./' + ODIR + '/tokenizer_bagofwords.json', 'w', encoding='utf-8') as f:
		f.write(json.dumps(tok_json, ensure_ascii=False))
    
	return matrixes

X = tokenizeData(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

def CNN(max_words):
	inputs = Input(name='inputs', shape=(max_words,))

	dense1 = Dense(units=1024, activation='relu')(inputs)
	dense1_BN = BatchNormalization()(dense1)
	dropOut1 = Dropout(0.5)(dense1_BN)

	dense2 = Dense(units=2048, activation='relu')(dropOut1)
	dense2_BN = BatchNormalization()(dense2)
	dropOut2 = Dropout(0.5)(dense2_BN)

	dense3 = Dense(units=2048, activation='relu')(dropOut2)
	dense3_BN = BatchNormalization()(dense3)
	dropOut3 = Dropout(0.5)(dense3_BN)

	dense4 = Dense(units=1024, activation='relu')(dropOut3)
	dense4_BN = BatchNormalization()(dense4)
	dropOut4 = Dropout(0.5)(dense4_BN)

	global Y_CLASS
	output = Dense(units=Y_CLASS, activation='softmax')(dropOut4)
	model = Model(inputs=inputs, outputs=output)
	
	return model

model = CNN(MAX_WORDS)
model.summary()
plot_model(model, to_file='./'+ ODIR + '/model.png')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(	X_train,
			Y_train,
			batch_size=200,
			epochs=1024,
			validation_split=0.2,
			callbacks = [	EarlyStopping(	monitor='val_loss',
							patience=5,
						)])

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