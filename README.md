# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the libraries and read the data frame using pandas.

2.Calculate the null values present in the dataset and apply label encoder.

3.Determine test and training data set and apply decison tree regression in dataset.

4.Calculate Mean square error,data prediction and r2.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: DHARSAN KUMAR R
RegisterNumber:  212223240028
*/
```
```
import pandas as pd
data=pd.read_csv("C:/Users/admin/Downloads/Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['Position']=le.fit_transform(data['Position'])
data.head()
x=data[['Position','Level']]
y=data['Salary']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_predict)
mse
r2=metrics.r2_score(y_test,y_predict)
r2
dt.predict([[5,6]])
```
## Output:
### head()
![image](https://github.com/SanjayBalaji0/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/145533553/ccd14a14-4f7b-4345-bdf0-556fb8f6f55e)
### info()
![image](https://github.com/SanjayBalaji0/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/145533553/25eca5c6-47ca-473d-a75b-1893eb19168a)
### isnull().sum()
![image](https://github.com/SanjayBalaji0/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/145533553/5ac6223d-1952-46fa-b4ea-81dc00c532a1)
### mse
![image](https://github.com/SanjayBalaji0/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/145533553/b5415f0a-35e9-4d6b-8739-a9932c765ff7)
### r2
![image](https://github.com/SanjayBalaji0/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/145533553/ada1ff86-7ff4-45f0-9feb-611cb0ecb17f)

### Predict
![image](https://github.com/SanjayBalaji0/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/145533553/7fdd67de-23cd-481e-853c-f9d7d45d5196)




## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
