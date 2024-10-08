# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: PRIYAADARSHINI.K
RegisterNumber: 212223240126 
*/
```
```
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
dataset=pd.read_csv('student_scores.csv')
print("dataset.head()")
print(dataset.head())
print("dataset.tail()")
print(dataset.tail())
dataset.info()
#assigning hours to X & scores to Y
print("X & Y values")
X=dataset.iloc[:,:-1].values
print(X)
Y=dataset.iloc[:,-1].values
print(Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
X_train.shape
X_test.shape
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)
print("Prediction values of X & Y")
Y_pred=reg.predict(X_test)
print(Y_pred)
print(Y_test)
plt.scatter(X_train,Y_train,color="pink")
plt.plot(X_train,reg.predict(X_train),color="brown")
plt.title('Training set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="blue")
plt.plot(X_test,reg.predict(X_test),color="silver")
plt.title('Test set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)

```
## Output:

![output1 (3)](https://github.com/user-attachments/assets/bc5493a5-acaf-496d-9473-2293b7759bd6)

![output2 (3)](https://github.com/user-attachments/assets/897468e4-42e7-49c7-89a3-7fd7a67abc6e)

![output3 (2)](https://github.com/user-attachments/assets/904ef7c4-4406-4481-b94b-f2e6fca46f56)

![output4](https://github.com/user-attachments/assets/43e9305a-a7f6-463a-9f61-db4c73870530)

![output5](https://github.com/user-attachments/assets/9a77cfe3-e70c-4f6b-b13f-95c6e6fc65b7)

![output6](https://github.com/user-attachments/assets/2e91b574-5340-49d8-87b0-4b9b2e57d09f)

![output7 (2)](https://github.com/user-attachments/assets/c158d3f1-cb4b-4a02-b002-84c3d71404c3)

![output8](https://github.com/user-attachments/assets/d76cfcd6-3003-41eb-8074-1e5aaf7b2371)

![output9](https://github.com/user-attachments/assets/102c34c5-f070-4fd5-bf86-6f3d50850d78)

![output10](https://github.com/user-attachments/assets/b0e5721d-be97-464e-821e-858fbf91ea35)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
