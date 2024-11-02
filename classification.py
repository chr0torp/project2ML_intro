import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.linear_model
import sklearn.model_selection
import sklearn.neighbors
import pandas as pd
from ucimlrepo import fetch_ucirepo
from dtuimldmtools import mcnemar

# Load Matlab data file and extract variables of interest
glass_identification = fetch_ucirepo(id=42)
X = glass_identification.data.features.to_numpy()
y = np.ravel(glass_identification.data.targets.to_numpy())

# Standardize data based on training set
mu = np.mean(X, 0)
sigma = np.std(X, 0)
X = (X - mu) / sigma

attribute_names = glass_identification.data.headers.to_numpy()

class_names = [
    "building_windows_float_processed",
    "building_windows_non_float_processed",
    "vehicle_windows_float_processed",
    "vehicle_windows_non_float_processed", # not used as no data is of this type
    "containers",
    "tableware",
    "headlamps"
]

N, M = X.shape
C = len(class_names)

## Crossvalidation
# Create crossvalidation partition for evaluation
K_outer = 10
CV_outer = sklearn.model_selection.KFold(n_splits=K_outer,shuffle=True)
K_inner = 10
CV_inner = sklearn.model_selection.KFold(n_splits=K_inner,shuffle=True)

# Initialize variables 
final_lambdas = np.empty(K_outer)
train_error = np.empty(K_outer)
test_error = np.empty(K_outer)

# Parameter lambda for method 1 multinomial regression
n_lambdas = 200
lambdas = np.power(10, np.linspace(-6, 2, n_lambdas))

# Parameter k for method 2 k nearest neighbours
n_k = 100

# Arrays for statistical evaluation
y_hat_1 = []
y_hat_2 = []
y_hat_3 = []
y_true = []

# Start k-fold cross validation outer loop
for k_outer, (train_index_outer, test_index_outer) in enumerate(CV_outer.split(X)):
    # Extract training and test set for current CV fold
    X_train_outer = X[train_index_outer,:]
    y_train_outer = y[train_index_outer]
    X_test_outer = X[test_index_outer,:]
    y_test_outer = y[test_index_outer]

    method1_test_errors = np.zeros(n_lambdas)
    method2_test_errors = np.zeros(n_k)

    # Start k-fold cross validation inner loop
    for k_inner, (train_index_inner, test_index_inner) in enumerate(CV_inner.split(X_train_outer)):
        X_train_inner = X_train_outer[train_index_inner,:]
        y_train_inner = y_train_outer[train_index_inner]
        X_test_inner = X_train_outer[test_index_inner,:]
        y_test_inner = y_train_outer[test_index_inner]
        
        # %% Model fitting and prediction
        # Standardize data based on training set
        mu = np.mean(X_train_inner, 0)
        sigma = np.std(X_train_inner, 0)
        X_train_inner = (X_train_inner - mu) / sigma
        X_test_inner = (X_test_inner - mu) / sigma

        # Train models for each lambda value
        for l in range(len(lambdas)):
            # Fit multinomial logistic regression model
            mdl1 = sklearn.linear_model.LogisticRegression(
                solver="lbfgs",
                penalty="l2",
                C=1/lambdas[l],
                max_iter=5000
            )
            mdl1.fit(X_train_inner, y_train_inner)
            y_test_est_inner = mdl1.predict(X_test_inner)
            test_error_rate_inner = np.sum(y_test_est_inner != y_test_inner) / len(y_test_inner)
            method1_test_errors[l] += test_error_rate_inner

        # Train models for each k value
        for k in range(n_k):
            # Fit k nearest neighbours
            mdl2 = sklearn.neighbors.KNeighborsClassifier(
                n_neighbors=k+1,
                p=2,
                metric="minkowski",
                metric_params={}
            )
            mdl2.fit(X_train_inner, y_train_inner)
            y_test_est_inner = mdl2.predict(X_test_inner)
            test_error_rate_inner = np.sum(y_test_est_inner != y_test_inner) / len(y_test_inner)
            method2_test_errors[k] += test_error_rate_inner

    # %% Model fitting and prediction
    # Standardize data based on training set
    mu = np.mean(X_train_outer, 0)
    sigma = np.std(X_train_outer, 0)
    X_train_outer = (X_train_outer - mu) / sigma
    X_test_outer = (X_test_outer - mu) / sigma

    # Method 1 multinomial best model
    best_lambda = np.argmin(method1_test_errors)
    final_lambdas[k_outer] = best_lambda

    # Fit multinomial logistic regression model
    mdl1 = sklearn.linear_model.LogisticRegression(
        solver="lbfgs",
        penalty="l2",
        C=1/lambdas[best_lambda],
        max_iter=5000
    )
    mdl1.fit(X_train_outer, y_train_outer)
    method1_y_test_est_outer = mdl1.predict(X_test_outer)
    method1_test_error_rate_outer = np.sum(method1_y_test_est_outer != y_test_outer) / len(y_test_outer)
    y_hat_1 = np.append(y_hat_1, method1_y_test_est_outer)

    # Method 2 best model
    best_k = np.argmin(method2_test_errors) + 1

    # Fit k nearest neighbours
    mdl2 = sklearn.neighbors.KNeighborsClassifier(
        n_neighbors=best_k,
        p=2,
        metric="minkowski",
        metric_params={}
    )
    mdl2.fit(X_train_outer, y_train_outer)
    method2_y_test_est_outer = mdl2.predict(X_test_outer)
    method2_test_error_rate_outer = np.sum(method2_y_test_est_outer != y_test_outer) / len(y_test_outer)
    y_hat_2 = np.append(y_hat_2, method2_y_test_est_outer)

    # Method 3 baseline
    values, counts = np.unique(y_test_outer, return_counts=True)
    method3_y_test_est_outer = np.full(len(y_test_outer), values[np.argmax(counts)])
    method3_test_error_rate_outer = np.sum(method3_y_test_est_outer != y_test_outer) / len(y_test_outer)
    y_hat_3 = np.append(y_hat_3, method3_y_test_est_outer)

    y_true = np.append(y_true, y_test_outer)

    print("Cross validation loop", k_outer + 1, ":")
    print("Method 1 multinomial regression - ", lambdas[best_lambda], method1_test_error_rate_outer)
    print("Method 2 k nearest neighbours - ", best_k, method2_test_error_rate_outer)
    print("Method 3 baseline - ", method3_test_error_rate_outer)

# Compute the Jeffreys interval
alpha = 0.05
[thetahat1, CI1, p1] = mcnemar(y_true, y_hat_1, y_hat_2, alpha=alpha)
[thetahat2, CI2, p2] = mcnemar(y_true, y_hat_2, y_hat_3, alpha=alpha)
[thetahat3, CI3, p3] = mcnemar(y_true, y_hat_1, y_hat_3, alpha=alpha)
print("Comparison between multinomial regression and k-nearest neighbours", thetahat1, " CI: ", CI1, "p-value", p1)
print("Comparison between k-nearest neighbours and baseline", thetahat2, " CI: ", CI2, "p-value", p2)
print("Comparison between multinomial regression and baseline", thetahat3, " CI: ", CI3, "p-value", p3)

# Get median of best lambdas
final_lambda = np.median(final_lambdas)

# Fit multinomial logistic regression model
final_mdl = sklearn.linear_model.LogisticRegression(
    solver="lbfgs",
    penalty="l2",
    C=1/final_lambda,
    max_iter=5000
)
final_mdl.fit(X, y)

print("Model weights:", final_mdl.coef_)

# Plot the importance of the features in training weights of model
avg_importance = np.mean(np.abs(final_mdl.coef_), axis=0)
feature_importance = pd.DataFrame({'Feature': attribute_names[1:10], 'Importance': avg_importance})
feature_importance = feature_importance.sort_values('Importance', ascending=True)
feature_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(10, 6))
plt.show()
