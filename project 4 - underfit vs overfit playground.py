import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

np.random.seed(42)

# Settings — same quadratic + moderate noise
n_samples = 150
noise_std = 0.7

true_function = lambda x: 1.5 * x**2 - 2 * x + 0.8

X = np.sort(np.random.uniform(-3, 3, n_samples)).reshape(-1, 1)
y = true_function(X) + np.random.normal(0, noise_std, n_samples).reshape(-1, 1)

# Split: train (70%) + validation/test (30%)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Two models to compare
models = {
    "Underfit (Linear)": LinearRegression(),
    "Overfit (Poly degree 12)": make_pipeline(PolynomialFeatures(12), LinearRegression())
}

plt.figure(figsize=(14, 6))

for i, (name, model) in enumerate(models.items(), 1):
    # Train
    model.fit(X_train, y_train.ravel())
    
    # Predictions on train & validation
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    
    train_mse = mean_squared_error(y_train, y_pred_train)
    val_mse = mean_squared_error(y_val, y_pred_val)
    
    # Smooth curve for plotting
    X_plot = np.linspace(-3, 3, 300).reshape(-1, 1)
    y_plot = model.predict(X_plot)
    
    # Plot
    plt.subplot(1, 2, i)
    plt.scatter(X_train, y_train, color='blue', alpha=0.6, s=40, label='Train data')
    plt.scatter(X_val, y_val, color='orange', alpha=0.6, s=40, label='Validation data')
    plt.plot(X_plot, y_plot, color='red', linewidth=2.5, label=name)
    plt.plot(X_plot, true_function(X_plot), color='green', linestyle='--', linewidth=2, label='True function')
    
    plt.title(f"{name}\nTrain MSE: {train_mse:.2f} | Val MSE: {val_mse:.2f}")
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-3.2, 3.2)
    plt.ylim(-5, 10)

plt.tight_layout()
plt.show()

# Summary print for quick comparison
print("Underfit vs Overfit Summary:")
print(f"Linear model:       Train MSE = {mean_squared_error(y_train, models['Underfit (Linear)'].predict(X_train)):.2f} | Val MSE = {mean_squared_error(y_val, models['Underfit (Linear)'].predict(X_val)):.2f}")
print(f"Poly degree 12:     Train MSE = {mean_squared_error(y_train, models['Overfit (Poly degree 12)'].predict(X_train)):.2f} | Val MSE = {mean_squared_error(y_val, models['Overfit (Poly degree 12)'].predict(X_val)):.2f}")
print("\n→ Underfit: both errors high & similar")
print("→ Overfit: train error low, val error much higher")