import json
import numpy as np
import keras
from keras_preprocessing import text

with open('./tokenizer2.json') as f:
	data = json.load(f)
	tokenizer = text.tokenizer_from_json(data)
	#reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
	while(True):
		inText = input("press the input : ").split(' ')
		print(inText)
		res = [[tokenizer.word_index[w] for w in inText ]]
		print( res )
print('Load Tokenizer ...') 

