import numpy as np

layer_outputs = [4.8, 1.21, 2.385]

exp_values = np.exp(layer_outputs)
print('exponentiated values:')
print(exp_values)

norm_values = exp_values / np.sum(exp_values)
print('normalized exponentiated values:')
print(norm_values)

print('sum of normalized values:', np.sum(norm_values))
