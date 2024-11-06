import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as lm

# Fetch dataset
glass_identification = fetch_ucirepo(id=42)

# Data (as pandas dataframes)
x = glass_identification.data.features
target = glass_identification.data.targets
attributeNames = ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]

#EX 1

# Transform data to achieve that each column has mean 0 and standard deviation 1.
scaler = StandardScaler()
x_trans = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

# Check that the transformation is correct
mean_x = x_trans.mean()
std_x = x_trans.std(ddof=1)
print("Means (should be ~0):", mean_x)
print("Standard deviations (should be ~1):", std_x)

# Select the target variable and predictors
predict_idx = attributeNames.index("RI")
y = x_trans.iloc[:, predict_idx]

X_cols = list(range(0, predict_idx)) + list(range(predict_idx + 1, len(attributeNames)))
#X = x_trans.iloc[:, [1, 2, 3, 5, 6]]
X = x_trans.iloc[:, X_cols]

# Fit ordinary least squares regression model
model = lm.LinearRegression()
model.fit(X, y)

# Predict and compute residuals
y_est = model.predict(X)
residual = y_est - y

# Display plots
fig, (ax1, ax2) = plt.subplots(1, 2)
# Add the trend line
max_value = max(y.max(), y_est.max())
min_value = min(y.min(), y_est.min())
ax1.set_xlabel("RI (true)") 
ax1.set_ylabel("RI (estimated)")
ax1.plot([min_value, max_value], [min_value, max_value], color='red', linestyle='--', label='Trend line')
ax1.legend()
ax1.plot(y, y_est, ".g")
ax2.hist(residual, 40)
ax2.set_xlabel("Residual")
ax2.set_title('Residual histogram')
plt.show()
