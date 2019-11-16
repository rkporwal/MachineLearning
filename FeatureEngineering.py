# importing essentials libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing dataset file
df=pd.read_csv("FeatureEngineering.csv")

df.info()

df=df.fillna(0)

df.info()
df.describe()
df['MSSubClass'].value_counts()

df.hist(bins=50,figsize=(20,20))

corr_matrix=df.corr()
corr_matrix['MSSubClass'].sort_values(ascending=False)

from pandas.plotting import scatter_matrix
attributes=['MSSubClass','2ndFlrSF']
scatter_matrix(df[attributes],figsize=(20,20))
df.plot(kind='scatter',x='MSSubClass',y='2ndFlrSF',alpha=0.8)