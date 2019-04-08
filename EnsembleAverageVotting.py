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

EMB = tfjs.converters.load_keras_model('./1D_YoonKim_Model/model.json')
EMB.save('1D_YoonKim_Model_hyp.h5')
EMB_MODEL = load_model('1D_YoonKim_Model_hyp.h5')

BOW = tfjs.converters.load_keras_model('./DNN_NE_HypOpt/model.json')
BOW.save('DNN_NE_hyp.h5')
BOW_MODEL = load_model('DNN_NE_hyp.h5')

CBOW = tfjs.converters.load_keras_model('./Transfer_CBOW_DNN/model.json')
CBOW.save('Transfer_CBOW_DNN_hyp.h5')
CBOW_MODEL = load_model('Transfer_CBOW_DNN_hyp.h5')

EMB_MODEL.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
BOW_MODEL.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
CBOW_MODEL.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

def ensemble():
    global BOW_MODEL, EMB_MODEL, CBOW_MODEL

    inputA = Input(shape=(MAX_LEN,))
    inputB = Input(shape=(MAX_WORD,))

    #average = CBOW_MODEL(inputA)
    # ALL
    # average = Average()([EMB_MODEL(inputA), CBOW_MODEL(inputA), BOW_MODEL(inputB)])
    # EMB, CBOW
    # average = Average()([EMB_MODEL(inputA), CBOW_MODEL(inputA)])
    # EMB, BOW
    # average = Average()([EMB_MODEL(inputA), BOW_MODEL(inputB)])
    # CBOW, BOW
    average = Average()([CBOW_MODEL(inputA), BOW_MODEL(inputB)])

    model = Model(inputs=[inputA, inputB], outputs=average)
    return model


model_ensemble = ensemble()
model_ensemble.summary()
model_ensemble.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

score = model_ensemble.evaluate([X_testA, X_testB], Y_testA)

print('Test loss: ', score[0])
print('Test Accuracy: ', score[1])
