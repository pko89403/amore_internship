
jsonList = ['./Input_json/train.json', './Input_json/test.json']

import pandas as pd;
import numpy as np;

for i in jsonList:
    jTmp = pd.read_json(i, orient='column')

    # preprocessing
    jTmp = jTmp.drop('id',1)
    jTmp['ingredients'] = [' '.join(map(str, l)) for l in jTmp['ingredients']]

    jTmp.to_csv(i + '.csv', header=True, sep=',', index=False)





