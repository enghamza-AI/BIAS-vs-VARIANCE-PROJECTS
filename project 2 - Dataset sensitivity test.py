#Project 2: Dataset Sensitivity Test

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

np.random.seed(42)

# Shared settings
n_samples = 120
degree = 8                  # moderately complex → enough to show variance
noise_std = 0.7

true_function = lambda x: 1.5 * x**2 - 2 * x + 0.8

# Generate base X (same for both datasets)
X = np.sort(np.random.uniform(-3, 3, n_samples)).reshape(-1, 1)

# Dataset A: original noisy version
y_A = true_function(X) + np.random.normal(0, noise_std, n_samples).reshape(-1, 1)

# Dataset B: slightly modified (add tiny extra noise + small systematic shift)
y_B = true_function(X) + np.random.normal(0, noise_std, n_samples).reshape(-1, 1) + 0.15  # small bias shift

# Function to train model and get predictions
def train_and_predict(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    poly = PolynomialFeatures(degree)
    model = make_pipeline(poly, LinearRegression())
    model.fit(X_train, y_train.ravel())
    
    # Predict on a fine grid for smooth curve
    X_grid = np.linspace(-3, 3, 300).reshape(-1, 1)
    y_grid_pred = model.predict(X_grid)
    
    # Also compute train/test MSE for reference
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    
    return X_grid, y_grid_pred, train_mse, test_mse

# Train on both datasets
X_grid, y_A_pred, train_mse_A, test_mse_A = train_and_predict(X, y_A)
_, y_B_pred, train_mse_B, test_mse_B = train_and_predict(X, y_B)

# Plot everything
plt.figure(figsize=(12, 7))

# Data points
plt.scatter(X, y_A, color='blue', alpha=0.5, s=40, label='Dataset A (original)')
plt.scatter(X, y_B, color='orange', alpha=0.5, s=40, label='Dataset B (slight shift + noise)')

# True function
plt.plot(X_grid, true_function(X_grid), color='green', linestyle='--', linewidth=2, label='True function')

# Model fits
plt.plot(X_grid, y_A_pred, color='blue', linewidth=2.5, label=f'Model A (deg {degree})')
plt.plot(X_grid, y_B_pred, color='orange', linewidth=2.5, label=f'Model B (deg {degree})')

plt.title(f'Dataset Sensitivity Test\nSame model (poly degree {degree}) on slightly different data\n'
          f'Train MSE A: {train_mse_A:.2f} | Test MSE A: {test_mse_A:.2f}\n'
          f'Train MSE B: {train_mse_B:.2f} | Test MSE B: {test_mse_B:.2f}')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(-3.2, 3.2)
plt.ylim(-5, 10)
plt.show()

# Compute difference between the two predictions
pred_diff = np.abs(y_A_pred - y_B_pred)
mean_diff = np.mean(pred_diff)
max_diff = np.max(pred_diff)

print(f"\nMean absolute difference between Model A and Model B predictions: {mean_diff:.3f}")
print(f"Max difference: {max_diff:.3f}")
print("→ Larger difference = higher variance (model is sensitive to small data changes)")