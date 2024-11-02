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
    rlr_validate,
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


K = 2
KF = KFold(K, shuffle=True)
base_values_outer = []
Lmodel_values_outer = []
ANN1_values_outer = []
ANN3_values_outer = []

# Define the model structure
# n_hidden_units = 3  # number of hidden units in the signle hidden layer
modelANN1 = lambda: torch.nn.Sequential(
    torch.nn.Linear(x.shape[1], 1),  # M features to H hiden units
    # 1st transfer function, either Tanh or ReLU:
    torch.nn.Tanh(),  # torch.nn.ReLU(),
    torch.nn.Linear(1, 1),  # H hidden units to 1 output neuron
    torch.nn.Sigmoid(),  # final tranfer function
)
modelANN3 = lambda: torch.nn.Sequential(
    torch.nn.Linear(x.shape[1], 3),  # M features to H hiden units
    # 1st transfer function, either Tanh or ReLU:
    torch.nn.Tanh(),  # torch.nn.ReLU(),
    torch.nn.Linear(3, 1),  # H hidden units to 1 output neuron
    torch.nn.Sigmoid(),  # final tranfer function
)
loss_fn = torch.nn.MSELoss()
max_iter = 10000

# Outer cross-validation loop
for train_index_outer, test_index_outer in KF.split(x_train):
    x_train_outer, x_test_outer = x_train.iloc[train_index_outer], x_train.iloc[test_index_outer]
    y_train_outer, y_test_outer = y_train.iloc[train_index_outer], y_train.iloc[test_index_outer]

    # Inner cross-validation
    K_inner = 3  # Number of inner folds
    KF_inner = KFold(K_inner, shuffle=True)

    base_values_inner = []
    Lmodel_values_inner = []
    ANN1_values_inner = []
    ANN3_values_inner = []

    # Inner cross-validation loop
    for train_index_inner, val_index_inner in KF_inner.split(x_train_outer):
        x_train_inner, x_val_inner = x_train_outer.iloc[train_index_inner], x_train_outer.iloc[val_index_inner]
        y_train_inner, y_val_inner = y_train_outer.iloc[train_index_inner], y_train_outer.iloc[val_index_inner]


        # To numpy
        X_train_inner_np = x_train_inner.values
        y_train_inner_np = y_train_inner.values

        # Compute the baseline
        mean_y_train_inner = y_train_inner.mean()
        y_pred_baseline_inner = pd.Series(mean_y_train_inner, index=y_val_inner.index)
        base_inner = np.mean((y_val_inner - y_pred_baseline_inner)**2)
        base_values_inner.append(base_inner)

        # RLR model
        lambdas = np.power(10.0, range(-5, 9))
        opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(
            X_train_inner_np, y_train_inner_np, lambdas, K_inner
        )
        Lmodel_values_inner.append(opt_val_err)

        # ANN models
        X_train_tensor = torch.Tensor(x_train_inner.values)
        y_train_tensor = torch.Tensor(y_train_inner.values).reshape(-1, 1)
        X_val_tensor = torch.Tensor(x_val_inner.values)

        # ANN1
        net1, final_loss1, learning_curve1 = train_neural_net(
            modelANN1, loss_fn, X=X_train_tensor, y=y_train_tensor, n_replicates=3, max_iter=max_iter
        )
        y_val_pred_ANN1 = net1(X_val_tensor)
        y_val_pred_ANN1 = (y_val_pred_ANN1 > 0.5).type(torch.float).squeeze().numpy()
        ANN1_inner = np.mean((y_val_inner - y_val_pred_ANN1)**2)
        ANN1_values_inner.append(ANN1_inner)

        # ANN3
        net3, final_loss3, learning_curve3 = train_neural_net(
            modelANN3, loss_fn, X=X_train_tensor, y=y_train_tensor, n_replicates=3, max_iter=max_iter
        )
        y_val_pred_ANN3 = net3(X_val_tensor)
        y_val_pred_ANN3 = (y_val_pred_ANN3 > 0.5).type(torch.float).squeeze().numpy()
        ANN3_inner = np.mean((y_val_inner - y_val_pred_ANN3)**2)
        ANN3_values_inner.append(ANN3_inner)

    # Calculate average metrics for the inner loop
    base_values_outer.append(np.mean(base_values_inner))
    Lmodel_values_outer.append(np.mean(Lmodel_values_inner))
    ANN1_values_outer.append(np.mean(ANN1_values_inner))
    ANN3_values_outer.append(np.mean(ANN3_values_inner))





Lmodel_values = np.array(Lmodel_values_outer)
print("Linear Model values:", Lmodel_values)
Lmodel_baseline = np.mean(Lmodel_values)
print("average Linear Model:", Lmodel_baseline)

base_values = np.array(base_values_outer)
print("base values:", base_values)
base_baseline = np.mean(base_values)
print("average Baseline:", base_baseline)


ANN1_values = np.array(ANN1_values_outer)
print("ANN 1 values:", ANN1_values)
ANN1_avg = np.mean(ANN1_values)
print("average ANN 1 :", ANN1_avg)

ANN3_values = np.array(ANN3_values_outer) 
print("ANN 3 values:", ANN3_values)
ANN3_avg = np.mean(ANN3_values)
print("average ANN 3 :", ANN3_avg)
