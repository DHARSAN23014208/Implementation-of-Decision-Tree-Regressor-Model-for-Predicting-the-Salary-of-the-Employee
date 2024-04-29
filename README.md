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
![image](https://github.com/DHARSAN23014208/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/149365413/db1ef1ff-349f-48c1-a33e-37c8c4e30ce4)

### info()

![image](https://github.com/DHARSAN23014208/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/149365413/680437a9-5e35-4228-8bdf-83e6d5c90ae1)

### isnull().sum()

![image](https://github.com/DHARSAN23014208/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/149365413/ce8c97bf-7ead-4148-b684-fbeb6043aac8)

### mse

![image](https://github.com/DHARSAN23014208/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/149365413/70dd12d6-03f4-4cb1-ac5d-977aee6e9bd0)

### r2

![image](https://github.com/DHARSAN23014208/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/149365413/958e08be-e9b2-4179-bc87-ea13b8ba7201)

### Predict

![image](https://github.com/DHARSAN23014208/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/149365413/872ed18f-a9be-4afc-8703-c671117a6c48)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
