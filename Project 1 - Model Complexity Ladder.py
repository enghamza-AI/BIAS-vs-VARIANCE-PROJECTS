import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

np.random.seed(42)

# Generate data: true underlying = quadratic + noise
n_samples = 100
X = np.sort(np.random.uniform(-3, 3, n_samples)).reshape(-1, 1)
true_function = lambda x: 1.5 * x**2 - 2 * x + 0.8
y_true = true_function(X)
y = y_true + np.random.normal(0, 0.7, n_samples).reshape(-1, 1)  # moderate noise

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Different model complexities
degrees = [0, 1, 2, 5, 10, 15]  # constant, linear, quadratic, ..., very high
train_errors = []
test_errors = []
models = []

plt.figure(figsize=(15, 10))

for i, degree in enumerate(degrees, 1):
    # Create polynomial model
    poly = PolynomialFeatures(degree)
    model = make_pipeline(poly, LinearRegression())
    model.fit(X_train, y_train.ravel())
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    
    train_errors.append(train_mse)
    test_errors.append(test_mse)
    models.append(model)
    
    # Plot fit
    plt.subplot(2, 3, i)
    X_plot = np.linspace(-3, 3, 200).reshape(-1, 1)
    y_plot = model.predict(X_plot)
    
    plt.scatter(X_train, y_train, color='blue', s=30, alpha=0.6, label='Train')
    plt.scatter(X_test, y_test, color='orange', s=30, alpha=0.6, label='Test')
    plt.plot(X_plot, y_plot, color='red', linewidth=2, label=f'degree={degree}')
    plt.plot(X_plot, true_function(X_plot), color='green', linestyle='--', label='True')
    plt.title(f'Degree {degree}\nTrain MSE: {train_mse:.2f} | Test MSE: {test_mse:.2f}')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Plot error vs complexity
plt.figure(figsize=(8, 5))
plt.plot(degrees, train_errors, 'o-', color='blue', label='Train MSE')
plt.plot(degrees, test_errors, 'o-', color='red', label='Test MSE')
plt.xlabel('Model complexity (polynomial degree)')
plt.ylabel('Mean Squared Error')
plt.title('Bias-Variance Tradeoff: Error vs Complexity')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')  # easier to see the U-shape
plt.show()