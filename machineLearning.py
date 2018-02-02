#!/usr/bin/python

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import svm

df = pd.read_csv('ArmFlection.csv')
df.columns = ['X4','X3','X2','X1', 'Y']
df = df.drop(['X4','X1'], 1)
df.head()
X = df.values[:, 0:2]
Y = df.values[:, 2]

from sklearn.model_selection import train_test_split

support = svm.SVC()

trainX, testX, trainY, testY = train_test_split( X, Y, test_size = 0.20)

support.fit(trainX[:], trainY[:])
print('Accuracy: \n', support.score(testX, testY))


pred = support.predict(testX)
print(pred)
exit()

sns.set_context("notebook", font_scale=1.1)
sns.set_style("ticks")

sns.lmplot('X1','X2', scatter=True, fit_reg=False, data=df, hue='Y')
plt.ylabel('X2')
plt.xlabel('X1')