import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# tracking the position salaries
df=pd.read_csv('position_salaries.csv')

# printing to ensure it looks correct
print(df)

df.info()

df['Position'].value_counts()

plt.scatter(df['Level'],df['Salary'])

X=df.iloc[:,1].values.reshape(-1,1)
y=df.iloc[:,-1].values

X

y

# Splitting data set
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=0)
X_train
print(X_train.shape)
print(X_test.shape)
from sklearn.linear_model import LinearRegression

L=LinearRegression()
L.fit(X_train , y_train)

y_pred = L.predict(X_test)
from sklearn.metrics import r2_score , mean_squared_error , mean_absolute_error

print('The r2 score',r2_score(y_test,y_pred))
print('The rmse',np.sqrt(mean_squared_error(y_test,y_pred)))
print('The mean absolute error',mean_absolute_error(y_test,y_pred))

plt.plot(X_test,y_pred,color='red')
plt.scatter(X,y)
plt.legend()
plt.show()

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 5)
X_train_poly=poly.fit_transform(X_train)
X_test_poly=poly.fit_transform(X_test)
print(X_train_poly)

print(X_test_poly)
L.fit(X_train_poly,y_train)

y_pred_2 = L.predict(X_test_poly)
print('The r2 score',r2_score(y_test,y_pred_2))
print('The rmse',np.sqrt(mean_squared_error(y_test,y_pred_2)))
print('The mean absolute error',mean_absolute_error(y_test,y_pred_2))

poly_2 = PolynomialFeatures(degree = 10)
X_train_poly_2=poly_2.fit_transform(X_train)
X_test_poly_2=poly_2.fit_transform(X_test)
L.fit(X_train_poly_2,y_train)
y_pred_3 = L.predict(X_test_poly_2)

print('The r2 score',r2_score(y_test,y_pred_3))
print('The rmse',np.sqrt(mean_squared_error(y_test,y_pred_3)))
print('The mean absolute error',mean_absolute_error(y_test,y_pred_3))

poly_3 = PolynomialFeatures(degree = 8)
X_train_poly_3=poly_3.fit_transform(X_train)
X_test_poly_3=poly_3.fit_transform(X_test)
L.fit(X_train_poly_3,y_train)
y_pred_4 = L.predict(X_test_poly_3)

print('The r2 score',r2_score(y_test,y_pred_4))
print('The rmse',np.sqrt(mean_squared_error(y_test,y_pred_4)))
print('The mean absolute error',mean_absolute_error(y_test,y_pred_4))

poly_4 = PolynomialFeatures(degree = 7)
X_train_poly_4=poly_4.fit_transform(X_train)
X_test_poly_4=poly_4.fit_transform(X_test)
L.fit(X_train_poly_4,y_train)
y_pred_5 = L.predict(X_test_poly_4)

print('The r2 score',r2_score(y_test,y_pred_5))
print('The rmse',np.sqrt(mean_squared_error(y_test,y_pred_5)))
print('The mean absolute error',mean_absolute_error(y_test,y_pred_5))

poly_5 = PolynomialFeatures(degree = 6)
X_train_poly_5=poly_5.fit_transform(X_train)
X_test_poly_5=poly_5.fit_transform(X_test)
L.fit(X_train_poly_5,y_train)
y_pred_6 = L.predict(X_test_poly_5)

print('The r2 score',r2_score(y_test,y_pred_6))
print('The rmse',np.sqrt(mean_squared_error(y_test,y_pred_6)))
print('The mean absolute error',mean_absolute_error(y_test,y_pred_6))


plt.scatter(X_test,y_pred_6,color='red')
plt.scatter(X,y,marker='+',color='green')
plt.legend()
plt.show()