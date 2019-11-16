
# importing essentials libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing dataset file
df=pd.read_csv("StatisticsData.csv")

# calculating mean, median, mode for all features (or one )  in dataset
print(df.sample)
print(df.mean())  # for completed dataframe
print(df.median())
print(df.mode())
print(df.var())
print(df.loc[:,"sample"].var(axis=0)) #for specific column
print(df.std(axis=0))
print(df.loc[:,"sample"].std())
plt.boxplot(df['sample'])
plt.show()


