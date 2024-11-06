import numpy as np
import importlib_resources
import pandas as pd
from ucimlrepo import fetch_ucirepo 

# fold
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler

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
x = glass_identification.data.features.to_numpy()
y = np.ravel(glass_identification.data.targets.to_numpy())
y = y - 1
  
# print the data
# print(x)
# print(y)
# print(y.head())
# print(y.columns)

mu = np.mean(x, 0)
sigma = np.std(x, 0)
x = (x - mu) / sigma

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

K = 3  # number of folds
KF = KFold(K, shuffle=True)

baseline_errors = []
base_values_outer = []
Lmodel_values_outer = []
ANN_values_outer = []
optimal_hidden_units = []
optimal_lambdas = []
ann_errors = []
linear_errors = []
baseline_errors = []

lambdas = np.power(10.0, np.arange(1, 2.5, 0.1)) 
num_classes = 7

# Define the model structure
# n_hidden_units = 3  # number of hidden units in the signle hidden layer
modelANN = lambda: torch.nn.Sequential(
    torch.nn.Linear(x.shape[1], 1),  # M features to H hiden units
    # 1st transfer function, either Tanh or ReLU:
    torch.nn.Tanh(),  # torch.nn.ReLU(),
    torch.nn.Linear(1, num_classes),  # H hidden units to 1 output neuron
    torch.nn.Softmax(dim=1)  # final tranfer function
)

loss_fn = torch.nn.CrossEntropyLoss()
max_iter = 1000

# Outer cross-validation loop
for i, (train_index_outer, test_index_outer) in enumerate(KF.split(x_train)):
    print(f"Outer fold: {i+1}/{K}")

    x_train_outer, x_test_outer = x_train[train_index_outer], x_train[test_index_outer] 
    y_train_outer, y_test_outer = y_train[train_index_outer], y_train[test_index_outer]
 
    # Inner cross-validation
    K_inner = 2  # Number of inner folds
    KF_inner = KFold(K_inner, shuffle=True)

    base_values_inner = []
    Lmodel_values_inner = []
    ANN_values_inner = []
    Error_train_nofeatures = np.empty((K_inner, 1))
    train_error = np.empty((K_inner, len(lambdas)))
    val_error = np.empty((K_inner, len(lambdas)))
    optimal_lambdas_inner = []

    # Inner cross-validation loop
    for j, (train_index_inner, val_index_inner) in enumerate(KF_inner.split(x_train_outer)):
        print(f"  Inner fold: {j+1}/{K_inner}")

        x_train_inner, x_val_inner = x_train_outer[train_index_inner], x_train_outer[val_index_inner]  
        y_train_inner, y_val_inner = y_train_outer[train_index_inner], y_train_outer[val_index_inner] 

        mu_inner = np.mean(x_train_inner, 0)
        sigma_inner = np.std(x_train_inner, 0)
        x_train_inner = (x_train_inner - mu_inner) / sigma_inner
        x_val_inner = (x_val_inner - mu_inner) / sigma_inner

        # Compute the baseline
        values, counts = np.unique(y_train_inner, return_counts=True)
        base_inner = np.mean(y_train_inner)
        Error_train_nofeatures[j] = (np.square(y_train_inner - y_train_inner.mean()).sum(axis=0) / y_train_inner.shape[0])
        base_values_inner.append(Error_train_nofeatures[j])

        # Linear model
        w = np.empty((x_train_inner.shape[1], K_inner, len(lambdas)))
        Xty = x_train_inner.T @ y_train_inner
        XtX = x_train_inner.T @ y_train_inner
     
        for l in range(0, len(lambdas)):
            # Compute parameters for current value of lambda and current CV fold
            # note: "linalg.lstsq(a,b)" is substitue for Matlab's left division operator "\"
            lambdaI = lambdas[l] * np.eye(x_train_inner.shape[1])
            lambdaI[0, 0] = 0  # remove bias regularization
            w[:, j, l] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()
            # Evaluate training and test performance
            train_error[j, l] = np.power(y_train_inner - x_train_inner @ w[:, j, l].T, 2).mean(axis=0)
            val_error[j, l] = np.power(y_val_inner - x_val_inner @ w[:, j, l].T, 2).mean(axis=0) 

        # Find best value of lambda for current CV fold
        opt_lambda_idx = np.argmin(np.mean(val_error[j, :]))
        opt_lambda = lambdas[opt_lambda_idx]
        opt_val_err = np.min(val_error[j, :])
        Lmodel_values_inner.append(opt_val_err) 
        optimal_lambdas_inner.append(opt_lambda)

        # ANN models
        X_train_tensor = torch.Tensor(x_train_inner)
        y_train_tensor = torch.Tensor(y_train_inner).long() 
        X_val_tensor = torch.Tensor(x_val_inner)

        # ANN
        best_ann1_accuracy = 0
        best_n_hidden_units_ann1 = 1
        for n_hidden_units in [1, 30, 80, 150]: 
            modelANN = lambda: torch.nn.Sequential(
                torch.nn.Linear(x.shape[1], n_hidden_units), 
                torch.nn.Tanh(), 
                torch.nn.Linear(n_hidden_units, num_classes), 
                torch.nn.Softmax(dim=1)
            )

            net1, final_loss, learning_curve = train_neural_net(
                modelANN, loss_fn, X=X_train_tensor, y=y_train_tensor, n_replicates=3, max_iter=max_iter
            )
            y_val_pred_ANN = net1(X_val_tensor)
            y_val_pred_classes = np.argmax(y_val_pred_ANN.detach().numpy(), axis=1)
            
            ann_error_inner = np.mean(y_val_inner != y_val_pred_classes) 

            accuracy_ann = np.mean(y_val_inner == y_val_pred_classes)  
            if accuracy_ann > best_ann1_accuracy:
                best_ann_accuracy = accuracy_ann
                best_n_hidden_units_ann1 = n_hidden_units

        ANN_values_inner.append(ann_error_inner) 

    best_lambda_outer = np.mean(optimal_lambdas_inner)
    best_linear_model = lm.Ridge(alpha=best_lambda_outer).fit(x_train_outer, y_train_outer)
    linear_error_outer = np.power(y_test_outer - best_linear_model.predict(x_test_outer), 2).mean(axis=0)
    optimal_lambdas.append(np.mean(Lmodel_values_inner))
    linear_errors.append(linear_error_outer) 

    best_ann_model = lambda: torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], best_n_hidden_units_ann1), 
        torch.nn.Tanh(), 
        torch.nn.Linear(best_n_hidden_units_ann1, num_classes), 
        torch.nn.Softmax(dim=1)
    )

    X_train_outer_tensor = torch.Tensor(x_train_outer)
    y_train_outer_tensor = torch.Tensor(y_train_outer).long()
    X_test_outer_tensor = torch.Tensor(x_test_outer)

    net_outer, final_loss_outer, learning_curve_outer = train_neural_net(best_ann_model, loss_fn, X=X_train_outer_tensor, y=y_train_outer_tensor, n_replicates=3, max_iter=max_iter)

    y_test_pred_ANN = net_outer(X_test_outer_tensor)
    y_test_pred_classes = np.argmax(y_test_pred_ANN.detach().numpy(), axis=1)
    ann_error_outer = np.mean(y_test_outer != y_test_pred_classes) 
    ann_errors.append(ann_error_outer)

    # Calculate average metrics for the inner loop
    base_values_outer.append(np.mean(base_values_inner))
    Lmodel_values_outer.append(np.mean(Lmodel_values_inner))
    ANN_values_outer.append(np.mean(ANN_values_inner))

    optimal_hidden_units.append(best_n_hidden_units_ann1)
    ann_errors.append(np.mean(ANN_values_inner))
    baseline_errors.append(np.mean(base_values_inner))


print("Outer Fold|Optimal Hidden Units|ANN Error|Optimal Lambda|Linear Error|Baseline Error")
print("-" * 100) 
for i in range(K):
    print(f"{i+1:9d}|{optimal_hidden_units[i]:19d}|{ann_errors[i]:9.4f}|{optimal_lambdas[i]:14.4f}|{linear_errors[i]:12.4f}|{baseline_errors[i]:14.4f}")

# print("Outer Fold|Optimal Lambda|Linear Error|test|Baseline Error")
# print("-" * 100) 
# for i in range(K):
#     print(f"{i+1:9d}||{optimal_lambdas[i]:14.4f}|{linear_errors[i]:12.4f}|{baseline_errors[i]:14.4f}")


Lmodel_values = np.array(linear_errors)
Lmodel_lamda = np.array(optimal_lambdas)
print("Linear Model values:", Lmodel_values)
print("Linear Model lambda:", Lmodel_lamda)
Lmodel = np.mean(Lmodel_values)
Lmodel_lamda = np.mean(Lmodel_lamda)
# print("average Linear Model:", Lmodel)
# print("average Linear Model lambda:", Lmodel_lamda)

base_values = np.array(base_values_outer)
print("base values:", base_values)
base_baseline = np.mean(base_values)
print("average Baseline:", base_baseline)


# ANN_values = np.array(ANN_values_outer)
# print("ANN values:", ANN_values)
# ANN_avg = np.mean(ANN_values)
# print("average ANN :", ANN_avg)