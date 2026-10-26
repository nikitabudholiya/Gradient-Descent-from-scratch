# Gradient Descent

A simple guide to understanding Gradient Descent, the optimization technique behind most machine learning and deep learning models.

---

## What is Gradient Descent?

**Gradient Descent is an iterative optimization algorithm that minimizes a loss function by repeatedly moving the parameters in the direction opposite to the gradient.**

In simple words: the model makes a prediction, we measure how wrong it is (the loss), and we adjust the parameters a little bit to make it less wrong. We repeat this many times until the loss stops decreasing.

Think of it as walking down a hill in the fog. You can't see the bottom, but you can feel which way the ground slopes. You take a small step downhill, again and again, until you reach the lowest point.

---

## The Update Rule

$$\theta_{new} = \theta_{old} - \eta \cdot \nabla_\theta J(\theta)$$

| Symbol | Meaning |
|--------|---------|
| θ | Parameters of the model (weights / coefficients) |
| η (eta) | Learning rate (the step size) |
| J(θ) | Loss function |
| ∇J(θ) | Gradient of the loss with respect to the parameters (the slope; a vector of partial derivatives when there are many parameters) |

The gradient points uphill, so we subtract it to move downhill.

---

## Example: Linear Regression

Model: ŷ = wx + b

Loss (Mean Squared Error):

$$J(w, b) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

Gradients (using partial derivatives):

$$\frac{\partial J}{\partial w} = -\frac{2}{n} \sum x_i (y_i - \hat{y}_i)$$

$$\frac{\partial J}{\partial b} = -\frac{2}{n} \sum (y_i - \hat{y}_i)$$

Updates:

$$w = w - \eta \cdot \frac{\partial J}{\partial w}$$

$$b = b - \eta \cdot \frac{\partial J}{\partial b}$$

---

## Steps of the Algorithm

1. Start with random (or zero) values for the parameters.
2. Make predictions and calculate the loss.
3. Calculate the gradient of the loss with respect to each parameter.
4. Update the parameters: `parameter = parameter - learning_rate * gradient`
5. Repeat steps 2 to 4 until the loss stops decreasing.

---

## Why Do We Need Gradient Descent?

Linear regression can be solved directly using OLS (closed form):

$$\theta = (X^T X)^{-1} X^T y$$

This needs a matrix inverse, which costs about O(d³) where d is the number of features. It becomes very expensive when there are many features, and it does not exist at all for models like neural networks.

Gradient descent only needs gradients, so it:
- Scales to large data and many features
- Works for any differentiable model
- Is the foundation of deep learning (backpropagation computes the gradients, gradient descent uses them to update the weights)

Note: for small linear regression problems, OLS can actually be faster. Gradient descent wins when the problem is large or has no closed-form solution.

---

## Learning Rate (η)

| Learning rate | What happens |
|---------------|--------------|
| Too high | Overshoots the minimum, oscillates, or even diverges |
| Too low | Moves very slowly, takes a very long time to converge |
| Just right | Reaches the minimum smoothly and quickly |

There is no universal best value. Common starting points are 0.1, 0.01 and 0.001, and the right one depends on the problem.

---

## Effect of Feature Scaling

- **Features on different scales:** the loss contours are stretched and elongated, so gradient descent zig-zags and converges slowly.
- **Features on the same scale (standardized or normalized):** the contours are close to circles, so gradient descent takes a direct path to the minimum and converges faster.

Always scale your features before using gradient descent.

---

## Types of Gradient Descent

| Type | Data used per update | Notes |
|------|----------------------|-------|
| Batch GD | The whole dataset | Stable but slow on big data |
| Stochastic GD (SGD) | 1 sample | Fast but noisy |
| Mini-batch GD | A small batch (e.g. 32, 64) | Best balance, the standard in practice |

---

## Python Implementation

```python
import numpy as np

# Create data: y = 2x + 1 with a little noise
np.random.seed(42)
X = np.random.rand(100)
y = 2 * X + 1 + np.random.randn(100) * 0.1

# Initialize parameters
w, b = 0.0, 0.0
lr = 0.1
epochs = 1000
n = len(X)

for i in range(epochs):
    y_pred = w * X + b
    error = y - y_pred

    # Gradients of MSE loss
    dw = -(2 / n) * np.sum(X * error)
    db = -(2 / n) * np.sum(error)

    # Update parameters
    w = w - lr * dw
    b = b - lr * db

    if i % 200 == 0:
        loss = np.mean(error ** 2)
        print(f"Epoch {i}: loss = {loss:.4f}")

print(f"Learned w = {w:.2f}, b = {b:.2f}")  # close to 2 and 1
```

---

## Limitations

- Finds a **local** minimum. For convex losses (like MSE in linear regression) this is also the global minimum, but for deep networks it may be a local minimum or a saddle point.
- Sensitive to the learning rate.
- Slow when features are not scaled.

---

## Quick Summary

> Compute the gradient, step against it by the learning rate, and repeat until the loss stops decreasing.

$$\theta_{new} = \theta_{old} - \eta \cdot \nabla_\theta J(\theta)$$
