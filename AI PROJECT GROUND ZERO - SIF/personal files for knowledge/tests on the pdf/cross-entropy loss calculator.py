import math

# An example output from the output layer of the neural network
softmax_output = [0.7, 0.1, 0.2]

# Ground truth
target_output = [1, 0, 0]

# Cross-entropy loss calculation
loss = -(math.log(softmax_output[0]) * target_output[0] +
         math.log(softmax_output[1]) * target_output[1] +
         math.log(softmax_output[2]) * target_output[2])

print(loss)
