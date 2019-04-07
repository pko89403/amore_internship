import tensorflowjs as tfjs
import pandas as pd

from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import *
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Average
from Data_proc import text2seq, text2matrix, MAX_WORD, Y_CLASS, TRAINING_PATH, MAX_LEN

X_trainA, X_testA, Y_trainA, Y_testA = text2seq()
X_trainB, X_testB, Y_trainB, Y_testB = text2matrix()


print(Y_testA[0], Y_testB[0])



DNN_NE = tfjs.converters.load_keras_model('./DNN_NE_HypOpt/model.json')
DNN_NE.save('DNN_NE_hyp.h5')
DNN_NE_MODEL = load_model('DNN_NE_hyp.h5')

MultiChannelCNN = tfjs.converters.load_keras_model('./MultiChannelCNN/model.json')
MultiChannelCNN.save('MultiChannelCNN_hyp.h5')
MultiChannelCNN_MODEL = load_model('MultiChannelCNN_hyp.h5')

Transfer_CBOW_CNN = tfjs.converters.load_keras_model('./Transfer_CBOW_CNN/model.json')
Transfer_CBOW_CNN.save('Transfer_CBOW_CNN_hyp.h5')
Transfer_CBOW_CNN_MODEL = load_model('Transfer_CBOW_CNN_hyp.h5')

MultiChannelCNN_MODEL.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
Transfer_CBOW_CNN_MODEL.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
DNN_NE_MODEL.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])



def ensemble():
    global MultiChannelCNN_MODEL, Transfer_CBOW_CNN_MODEL, DNN_NE_MODEL

    #inputA = Input(shape=(MAX_LEN,))
    inputB = Input(shape=(MAX_WORD,))
    # ALL
    # average = Average()([MultiChannelCNN_MODEL(inputA), Transfer_CBOW_CNN_MODEL(inputA), DNN_NE_MODEL(inputB)])
    # EMB, CBOW
    # average = Average()([MultiChannelCNN_MODEL(inputA), Transfer_CBOW_CNN_MODEL(inputA)])
    # EMB, BOW
    average = DNN_NE_MODEL(inputB)
    # CBOW, BOW
    # average = Average()([Transfer_CBOW_CNN_MODEL(inputA), DNN_NE_MODEL(inputB)])

    model = Model(inputs=inputB, outputs=average)
    return model


model_ensemble = ensemble()
model_ensemble.summary()
model_ensemble.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

score = model_ensemble.evaluate(X_testB, Y_testA)

print('Test loss: ', score[0])
print('Test Accuracy: ', score[1])
