import numpy as np
import importlib_resources
import pandas as pd
from ucimlrepo import fetch_ucirepo 

# fold
from sklearn.model_selection import train_test_split, KFold

# linear model
import sklearn.linear_model as lm

# ANN
import torch
from dtuimldmtools import (
    draw_neural_net,
    train_neural_net,
    visualize_decision_boundary,
)
  
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


K = 10
KF = KFold(K, shuffle=True)
base_values = []
Lmodel_values = []
ANN_values = []

# Define the model structure
n_hidden_units = 4  # number of hidden units in the signle hidden layer
modelANN = lambda: torch.nn.Sequential(
    torch.nn.Linear(x.shape[1], n_hidden_units),  # M features to H hiden units
    # 1st transfer function, either Tanh or ReLU:
    torch.nn.Tanh(),  # torch.nn.ReLU(),
    torch.nn.Linear(n_hidden_units, 1),  # H hidden units to 1 output neuron
    torch.nn.Sigmoid(),  # final tranfer function
)
loss_fn = torch.nn.MSELoss()
max_iter = 10000

for train_index, val_index in KF.split(x_train):
    # linear model and baseline
    x_train_k, x_val = x_train.iloc[train_index], x_train.iloc[val_index]
    y_train_k, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

    # Compute the baseline
    # Compute the mean of the training set
    mean_y_train_k = y_train_k.mean()
    y_pred_baseline = pd.Series(mean_y_train_k, index=y_val.index)
    # Compute mean squared error
    base = np.mean((y_val - y_pred_baseline)**2)
    base_values.append(base)

    # Compute the linear model
    model = lm.LinearRegression()
    model.fit(x_train_k, y_train_k)
    y_val_pred = model.predict(x_val)
    # Compute mean squared error
    Lmodel = np.mean((y_val - y_val_pred)**2)
    Lmodel_values.append(Lmodel)

    # ANN
    X_train_tensor = torch.Tensor(x_train_k.values)
    y_train_tensor = torch.Tensor(y_train_k.values).reshape(-1, 1)
    X_val_tensor = torch.Tensor(x_val.values)

    net, final_loss, learning_curve = train_neural_net(
        modelANN, loss_fn, X=X_train_tensor, y=y_train_tensor, n_replicates=3, max_iter=max_iter
    )


    y_val_pred_ANN = net(X_val_tensor)
    y_val_pred_ANN = (y_val_pred_ANN > 0.5).type(torch.float).squeeze().numpy()

    # Compute mean squared error
    ANN = np.mean((y_val - y_val_pred_ANN)**2)
    ANN_values.append(ANN)


Lmodel_values = np.array(Lmodel_values)
print("Linear Model values:", Lmodel_values)
Lmodel_baseline = np.mean(Lmodel_values)
print("average Linear Model:", Lmodel_baseline)

base_values = np.array(base_values)
print("base values:", base_values)
base_baseline = np.mean(base_values)
print("average Baseline:", base_baseline)

ANN_values = np.array(ANN_values)
print("ANN values:", ANN_values)
ANN_avg = np.mean(ANN_values)
print("average ANN:", ANN_avg)