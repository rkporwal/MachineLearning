# importing essentials libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# importing dataset file
df=pd.read_csv("Hiring.csv")

df.experience.fillna(0,inplace=True)
df.test_score.fillna(df.test_score.mean(),inplace=True)

X=df.iloc[:,:3]
# converting word to integer values
def word_to_int(word):
    word_dict={'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7
               ,'eight':8,'nine':9,'ten':10,'eleven':11,0:0}
    return word_dict[word]
    
X['experience']=X['experience'].apply(lambda x : word_to_int(x))

y=df.iloc[:,-1]

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X,y)

pickle.dump(model,open('model.pkl','wb'))
