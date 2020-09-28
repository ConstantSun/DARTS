import torch

# from operations import *
# print('jii')
# print(test_0(4))

a = torch.tensor([1, 2, 3, 4, 5, 6, 7]).type("torch.DoubleTensor")
b = torch.tensor([4, 4, 4, 4, 4, 4, 4,]).type("torch.DoubleTensor")

c = a.sub(0.6, b)

print(c)