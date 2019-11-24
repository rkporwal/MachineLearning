import pandas as pd
import  numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

data=pd.read_csv("CellPhone.csv")
X = data.iloc[:,0:20]  #independent columns
y = data.iloc[:,-1]    #target column i.e price range

