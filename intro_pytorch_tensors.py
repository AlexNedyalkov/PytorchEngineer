import torch
import numpy as np

# creating tensors
x = torch.ones(2,2)
y = torch.rand(2,2)
z = torch.tensor([2,3,4,5])

print('x')
print(x)
print(x.type(), x.size())

print('y')
print(y)

print('z')
print(z)

# tensor operations
print('summation')
print(x + y)
# inplace addition
x.add_(y)
print(x)

print('multiplication')
print(x * y)

# slicing
print('Slicing')
a = torch.rand(5,2)
print(a)
print(a[:,0])

# reshaping
s = torch.rand(4,4)
print(s)
t = s.view((2,8))
print(t)
u = s.view(-1,2)
print(u)

# from tensors to numpy
v = u.numpy()
print(v)

u.add_(1)
print(u)
print(v)

# from numpy to tensors
w = torch.from_numpy(v)
print(w)

if torch.cuda.is_available():
    print('Cuda')
else:
    print('No Cuda')