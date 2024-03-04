import numpy as np

def cost_function(X, y, theta):
    """
    Compute the cost function (mean squared error) for linear regression.
    
    Parameters:
    X : numpy array
        Input features (m x n).
    y : numpy array
        Target variable (m x 1).
    theta : numpy array
        Model parameters (n x 1).
        
    Returns:
    float
        Cost function value.
    """
    m = len(y)
    predictions = np.dot(X, theta)
    error = predictions - y
    cost = np.sum(error ** 2) / (2 * m)
    return cost

def gradient_descent(X, y, theta, alpha, num_iterations):
    """
    Perform gradient descent to minimize the cost function.
    
    Parameters:
    X : numpy array
        Input features (m x n).
    y : numpy array
        Target variable (m x 1).
    theta : numpy array
        Initial model parameters (n x 1).
    alpha : float
        Learning rate.
    num_iterations : int
        Number of iterations.
        
    Returns:
    numpy array
        Optimized model parameters.
    list
        Cost function history.
    """
    m = len(y)
    cost_history = []
    
    for _ in range(num_iterations):
        predictions = np.dot(X, theta)
        error = predictions - y
        gradient = np.dot(X.T, error) / m
        theta -= alpha * gradient
        cost = cost_function(X, y, theta)
        cost_history.append(cost)
    
    return theta, cost_history

# Generate sample data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Add bias term to input features
X_b = np.c_[np.ones((100, 1)), X]

# Initialize model parameters
theta = np.random.randn(2, 1)

# Set hyperparameters
alpha = 0.1
num_iterations = 1000

# Perform gradient descent
theta_optimized, cost_history = gradient_descent(X_b, y, theta, alpha, num_iterations)

# Print optimized parameters
print("Optimized parameters (theta):", theta_optimized)

# Plot cost function history
import matplotlib.pyplot as plt
plt.plot(cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Gradient Descent: Cost Function History')
plt.show()