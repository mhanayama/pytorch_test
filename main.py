#!/bin/python3

import torch
import numpy as np

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

print(data)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

print(x_np)

x_ones = torch.ones_like(x_data)

print(x_ones)

x_rand = torch.rand_like(x_data, dtype=torch.float)

print(x_rand)

shape = (2, 3, )
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(rand_tensor)
print(ones_tensor)
print(zeros_tensor)

tensor = torch.rand(3, 4)

print(tensor.shape)
print(tensor.dtype)
print(tensor.device)

if torch.cuda.is_available():
    tensor = tensor.to('cuda')
else:
    tensor = tensor.to('cpu')

tensor = torch.ones(4, 4)
tensor[:, 1] = 0
print(tensor)

