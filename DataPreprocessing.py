# importing essentials libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing dataset file
df=pd.read_csv("HomePrice.csv")

X=df.iloc[:,0:2].values  #ndarray
y=df.iloc[:,[2]].values   #ndarray

# Handling  missing values
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values="NaN",strategy="mean",axis=0)
imputer.fit(X[:,[1]])
X[:,[1]]=imputer.transform(X[:,[1]])


# Handling Categorical features
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le=LabelEncoder()
X[:,0]=le.fit_transform(X[:,0])
ohe=OneHotEncoder(categorical_features=[0])
X=ohe.fit_transform(X).toarray()

X=X[:,1:]

# splitting into train test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

