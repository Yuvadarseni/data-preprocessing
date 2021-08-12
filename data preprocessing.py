#Data preprocessing

#importing Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import dataset
dataset=pd.read_csv("Data_preprocessing.csv")
dataset.isnull().sum()  

#dependent and independent variable
x=dataset.iloc[:,0:3].values
y=dataset.iloc[:,3:4].values

#import imputer and fill the null values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean',fill_value=None)
imputer=imputer.fit(x[:,1:3])
x[:,1:3]= imputer.transform(x[:,1:3])

#import label encoder and onehot encoder for categorical data
from sklearn.preprocessing import LabelEncoder
Label_x = LabelEncoder()
x[:,0]=Label_x.fit_transform(x[:,0])
from sklearn.compose import ColumnTransformer
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0])],remainder='passthrough')
x=np.array(columnTransformer.fit_transform(x),dtype=np.str)
x=x[:,1:]
Label_y=LabelEncoder()
y=Label_y.fit_transform(y)

#split the data into train and test
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest= train_test_split(x,y,test_size=.20,random_state=0)


#feature scaling
from sklearn.preprocessing import StandardScaler
Sc_x= StandardScaler()
xtrain= Sc_x.fit_transform(xtrain)
xtest= Sc_x.transform(xtest)