import pandas as pd
from pandas import DataFrame

data = {'ID': ['A1', 'A2', 'A3', 'A4', 'A5'], 'X1': [1,2,3,4,5], 'X2':[3,4.5,3.2,4.0,3.5]}

data_df = DataFrame(data)

print(data_df)

data_df.to_csv('~/intern_challenge/data_df.csv')
