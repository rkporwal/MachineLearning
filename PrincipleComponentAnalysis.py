import pandas as pd
import numpy as np
import random as rd
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt



dataset=pd.read_csv("CellPhone.csv")
X=dataset.iloc[:,0:21].values

pca=PCA()
X=pca.fit_transform(X)
pca.explained_variance_ratio_