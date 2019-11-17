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


### Handling multiple categorical features

use_cols=['SibSp','Pclass','Age','Parch','Sex','Embarked','Survived']

dataset=pd.read_csv("titanic.csv",usecols=use_cols)

temp=pd.get_dummies(dataset['Sex'],drop_first=True)
newdf=pd.concat([temp,dataset],axis='columns')
temp=pd.get_dummies(dataset['Embarked'],drop_first=True)
newdf=pd.concat([temp,newdf],axis='columns')
newdf.drop(['Sex','Embarked'],axis=1,inplace=True)

newdf.isnull().sum()

newdf=newdf.fillna({'Age':newdf['Age'].mean()})
newdf.isnull().sum()

y=newdf['Survived']
X=newdf.drop('Survived',axis=1)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)


from sklearn.linear_model import  LogisticRegression
model=LogisticRegression()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score
metrix=confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred)
