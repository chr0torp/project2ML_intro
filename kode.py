import numpy as np
import importlib_resources
import pandas as pd
from ucimlrepo import fetch_ucirepo 

from sklearn.model_selection import train_test_split
  
# fetch dataset 
glass_identification = fetch_ucirepo(id=42) 
  
# data (as pandas dataframes) 
x = glass_identification.data.features 
y = glass_identification.data.targets 
  
# print the data
# print(x)
# print(y)
# print(y.head())
# print(y.columns)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

y_train = pd.to_numeric(y_train['Type_of_glass'])
y_test = pd.to_numeric(y_test['Type_of_glass'])
mean_y_train = y_train.mean()
y_pred_baseline = pd.Series(mean_y_train, index=y_test.index)
mse_baseline = np.mean((y_test - y_pred_baseline)**2)
print("Baseline MSE:", mse_baseline)