import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

def dataframe_to_array(dataframe):
    # Make a copy of the original dataframe
    dataframe1 = dataframe.copy(deep=True)
    # Extract input & outupts as numpy arrays
    x = dataframe1.iloc[:, 1:].to_numpy()
    y = dataframe1['label'].to_numpy()
    return x,y

dataset = pd.read_csv('Project_02\\train.csv')
x,y = dataframe_to_array(dataset)

x = x.astype(float)
X_test = x
model = LogisticRegression(solver = 'lbfgs')
model.fit(x,y)

print('test:',model.predict(X_test[0:10]))

print('score:',model.score(x, y))
print('coef_:',model.coef_)
print('intercept_:',model.intercept_)
