import numpy as np
import sklearn.linear_model as lm
from matplotlib.pylab import (figure, grid, legend, loglog, semilogx, show, subplot, title, xlabel, ylabel, scatter, xscale,plot,)
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
import pandas as pd
from ucimlrepo import fetch_ucirepo 

def rlr_validate(X, y, lambdas, X_train_outer, i, cvf=10):
    CV = model_selection.KFold(cvf, shuffle=True)
    M = X.shape[1]
    w = np.empty((M, cvf, len(lambdas)))
    train_error = np.empty((cvf, len(lambdas)))
    test_error = np.empty((cvf, len(lambdas)))
    gen_error = np.empty((len(lambdas)))
    gen_error_Mean = np.empty((len(lambdas)))
    f = 0
    y = y.squeeze()
    for train_index, test_index in CV.split(X, y):
        X_train_inner = X[train_index]
        y_train_inner = y[train_index]
        X_test_inner = X[test_index]
        y_test_inner = y[test_index]

        # Standardize the training and set set based on training set moments
        mu = np.mean(X_train_inner[:, 1:], 0)
        sigma = np.std(X_train_inner[:, 1:], 0)

        X_train_inner[:, 1:] = (X_train_inner[:, 1:] - mu) / sigma
        X_test_inner[:, 1:] = (X_test_inner[:, 1:] - mu) / sigma

        # precompute terms
        Xty = X_train_inner.T @ y_train_inner
        XtX = X_train_inner.T @ X_train_inner
        for l in range(0, len(lambdas)):
            # Compute parameters for current value of lambda and current CV fold
            # note: "linalg.lstsq(a,b)" is substitue for Matlab's left division operator "\"
            lambdaI = lambdas[l] * np.eye(M)
            lambdaI[0, 0] = 0  # remove bias regularization
            w[:, f, l] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()
            # Evaluate training and test performance
            train_error[f, l] = np.power(y_train_inner - X_train_inner @ w[:, f, l].T, 2).mean(
                axis=0
            )
            test_error[f, l] = np.power(y_test_inner - X_test_inner @ w[:, f, l].T, 2).mean(axis=0)
        f = f + 1

    a = 0
    while a < len(lambdas):
        gen_error[a] = len(X_test_inner)/len(X_train_outer)*sum(test_error[:,a])
        gen_error_Mean[a] = np.mean(test_error[:,a])
        a = a+1
        

    gen_error_S = np.mean(gen_error, axis=0)
    opt_val_err = np.min(gen_error, axis=0)
    opt_lambda = lambdas[np.argmin(gen_error, axis=0)]
    train_err_vs_lambda = np.mean(train_error, axis=0)
    gen_err_vs_lambda = np.mean(gen_error, axis=0)
    test_err_vs_lambda = np.mean(test_error, axis=0)
    
    mean_w_vs_lambda = np.squeeze(np.mean(w, axis=1))

    return (
        opt_val_err,
        opt_lambda,
        mean_w_vs_lambda,
        train_err_vs_lambda,
        test_err_vs_lambda,
        gen_error,
        test_error,
    )

glass_identification = fetch_ucirepo(id=42)

# Data (as pandas dataframes)
x = glass_identification.data.features
target = glass_identification.data.targets
attributeNames = ["Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]
# Transform data to achieve that each column has mean 0 and standard deviation 1.
scaler = StandardScaler()
x_trans = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

# Select the target variable and predictors
predict_idx = 0
y = x_trans.iloc[:, predict_idx]

X_cols = list(range(0, predict_idx)) + list(range(predict_idx + 1, len(attributeNames)+1))
X = x_trans.iloc[:, X_cols]
#to convert to the correct format ==> array
X = X.to_numpy()
y = y.to_numpy()
N, M = X.shape

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0], 1)), X), 1)
attributeNames = ["Offset"] + attributeNames
M = M + 1

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(K, shuffle=True)

# Values of lambda
exponents = np.linspace(-3, 8, num=50) 
lambdas = np.power(10.0, exponents)
#lambdas = np.power(10.0, range(-10, 10))

# Initialize variables
# T = len(lambdas)
Error_train = np.empty((K, 1))
E_gen_outer = np.empty((K, 1))
Error_test = np.empty((K, 1))
Error_train_rlr = np.empty((K, 1))
Error_test_rlr = np.empty((K, 1))
Error_train_nofeatures = np.empty((K, 1))
Error_test_nofeatures = np.empty((K, 1))
w_rlr = np.empty((M, K))
mu = np.empty((K, M - 1))
sigma = np.empty((K, M - 1))
w_noreg = np.empty((M, K))

k = 0
for train_index, test_index in CV.split(X, y):
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]

    internal_cross_validation = 10

    (
        opt_val_err,
        opt_lambda,
        mean_w_vs_lambda,
        train_err_vs_lambda,
        test_err_vs_lambda,
        gen_err_vs_lambda,
        test_error,
    ) = rlr_validate(X_train, y_train, lambdas, X_train, internal_cross_validation)

    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)

    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :]) / sigma[k, :]
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :]) / sigma[k, :]

    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train

    # Compute mean squared error without using the input data at all
    Error_train_nofeatures[k] = (
        np.square(y_train - y_train.mean()).sum(axis=0) / y_train.shape[0]
    )
    Error_test_nofeatures[k] = (
        np.square(y_test - y_test.mean()).sum(axis=0) / y_test.shape[0]
    )

    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0, 0] = 0  # Do no regularize the bias term
    w_rlr[:, k] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr[k] = (
        np.square(y_train - X_train @ w_rlr[:, k]).sum(axis=0) / y_train.shape[0]
    )
    Error_test_rlr[k] = (
        np.square(y_test - X_test @ w_rlr[:, k]).sum(axis=0) / y_test.shape[0]
    )

    # Estimate weights for unregularized linear regression, on entire training set
    w_noreg[:, k] = np.linalg.solve(XtX, Xty).squeeze()
    # Compute mean squared error without regularization
    Error_train[k] = (
        np.square(y_train - X_train @ w_noreg[:, k]).sum(axis=0) / y_train.shape[0]
    )
    Error_test[k] = (
        np.square(y_test - X_test @ w_noreg[:, k]).sum(axis=0) / y_test.shape[0]
    )

    # Display the results for the last cross-validation fold
    if k == K - 1:
        figure(k, figsize=(12, 8))
        subplot(1, 3, 1)
        semilogx(lambdas, mean_w_vs_lambda.T[:, 1:], ".-")  # Don't plot the bias term
        xlabel("Regularization factor")
        ylabel("Mean Coefficient Values")
        #legend("Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe")
        grid()

        subplot(1, 3, 2)
        title("Optimal lambda: 1e{0}".format(np.log10(opt_lambda)))
        loglog(
            lambdas, train_err_vs_lambda.T, "b.-", lambdas, test_err_vs_lambda.T, "r.-",lambdas, gen_err_vs_lambda.T, "g.-"
        )
        xlabel("Regularization factor")
        ylabel("Squared error (crossvalidation)")
        legend(["Train error", "Validation error", "Generalization error"])
        grid()
        subplot(1, 3, 3)
        # Graficar los puntos de train_error para cada valor de lambda
        for i in range(10):
            scatter(lambdas, test_error[i, :], color='black', alpha=0.5,marker='.', label="Test Error" if i == 0 else "")
        # Graficar la línea de gen_err_vs_lambda
        plot(lambdas, gen_err_vs_lambda, color='red', marker='.', linestyle='-', label="Gen. Error")
        xscale('log')  
        xlabel("Regularization factor")
        ylabel('Squared error (crossvalidation)')
        legend()
        grid(True)
        show()
        #results for optimal lambda
        index = np.where(lambdas == opt_lambda)[0][0]
        optimal_weights = mean_w_vs_lambda.T[index, :] #bias + atributes weights
        print(optimal_weights)
        print(index)

    k += 1

show()
