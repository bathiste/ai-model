import math
layer_outputs = [4.8, 1.21, 2.385]

E = 2.71828182846

exp_values = []
for output in layer_outputs:
    exp_values.append(E ** output)

print('exponentiated values:')
print(exp_values)

norm_base = sum(exp_values)
norm_values = []

for value in exp_values:
    norm_values.append(value / norm_base)

print('Normalized exponentiated values:')
print(norm_values)

print('Sum of normalized values:', sum(norm_values))
