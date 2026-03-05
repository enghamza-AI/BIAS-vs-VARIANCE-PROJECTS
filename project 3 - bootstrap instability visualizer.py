import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

np.random.seed(42)

# Settings
n_samples = 120
degree = 8                  # same as project 2 — shows variance clearly
n_bootstraps = 50           # how many resamples
noise_std = 0.7

true_function = lambda x: 1.5 * x**2 - 2 * x + 0.8

# Generate original data once
X = np.sort(np.random.uniform(-3, 3, n_samples)).reshape(-1, 1)
y = true_function(X) + np.random.normal(0, noise_std, n_samples).reshape(-1, 1)

# Grid for smooth prediction curves
X_grid = np.linspace(-3, 3, 300).reshape(-1, 1)

# Store all bootstrap predictions
bootstrap_preds = np.zeros((n_bootstraps, len(X_grid)))

plt.figure(figsize=(12, 7))

# Plot original data
plt.scatter(X, y, color='gray', alpha=0.4, s=30, label='Original data')

# True function
plt.plot(X_grid, true_function(X_grid), color='green', linestyle='--', linewidth=2.5, label='True function')

# Bootstrap loop
for i in range(n_bootstraps):
    # Resample with replacement (bootstrap)
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    X_boot = X[indices]
    y_boot = y[indices]
    
    # Train model
    poly = PolynomialFeatures(degree)
    model = make_pipeline(poly, LinearRegression())
    model.fit(X_boot, y_boot.ravel())
    
    # Predict on grid
    y_pred_grid = model.predict(X_grid)
    bootstrap_preds[i] = y_pred_grid
    
    # Plot this bootstrap fit (semi-transparent)
    plt.plot(X_grid, y_pred_grid, color='blue', alpha=0.08, linewidth=1.2)

# Plot mean prediction + ±1 std band
mean_pred = np.mean(bootstrap_preds, axis=0)
std_pred = np.std(bootstrap_preds, axis=0)

plt.plot(X_grid, mean_pred, color='darkblue', linewidth=3, label='Mean bootstrap prediction')
plt.fill_between(X_grid.ravel(), mean_pred - std_pred, mean_pred + std_pred,
                 color='blue', alpha=0.25, label='±1 std band (variance)')

plt.title(f'Bootstrap Instability Visualizer\n'
          f'Poly degree {degree}, {n_bootstraps} bootstraps\n'
          f'Mean std of predictions: {np.mean(std_pred):.3f}')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(-3.2, 3.2)
plt.ylim(-5, 12)
plt.show()

# Summary stats
print(f"Average width of ±1 std band across X: {np.mean(2 * std_pred):.3f}")
print("→ Wider band = higher variance (model predictions unstable across resamples)")