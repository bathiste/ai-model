import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from nnfs.datasets import vertical_data

nnfs.init()

# Dense layer
class Layer_Dense:
    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # Forward pass
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


# ReLU activation
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


# Softmax activation
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


# Common loss class
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


# Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Create ReLU activation
activation1 = Activation_ReLU()

# Create second Dense layer with 3 input features and 3 output values
dense2 = Layer_Dense(3, 3)

# Create Softmax activation
activation2 = Activation_Softmax()

# Create loss function
loss_function = Loss_CategoricalCrossentropy()

# Forward pass through first Dense layer
dense1.forward(X)

# Forward pass through ReLU activation
activation1.forward(dense1.output)

# Forward pass through second Dense layer
dense2.forward(activation1.output)

# Forward pass through Softmax activation
activation2.forward(dense2.output)

# Output of first 5 samples
print(activation2.output[:5])

# Calculate loss
loss = loss_function.calculate(activation2.output, y)
print('loss:', loss)

predictions = np.argmax(activation2.output, axis=1)

# Convert one-hot targets to class indices if needed
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)

# Calculate accuracy
accuracy = np.mean(predictions == y)

# Print accuracy
print('acc:', accuracy)

# Create dataset
X, y = vertical_data(samples=100, classes=3)

# Create model
dense1 = Layer_Dense(2, 3)          # first dense layer, 2 inputs
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)          # second dense layer, 3 inputs, 3 outputs
activation2 = Activation_Softmax()

# Create loss function
loss_function = Loss_CategoricalCrossentropy()

# Helper variables
lowest_loss = 9_999_999             # initial large value

best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()

best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()
