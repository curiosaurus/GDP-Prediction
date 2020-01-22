from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
from pandas import *

k=pd.read_csv("mtcars.csv")

print(k)

X=k['disp','hp','wt'].values
y=k['mpg']
k.shape

regressor=LinearRegression()
regressor.fit(X,Y)
regressor.predict([[180]])

plt.scatter(X,Y)
plt.plot(X,regressor.predict(X),color='red')
