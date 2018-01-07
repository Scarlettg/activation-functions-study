import torch
from torch.autograd import Variable

# This is fine and works as expected
t_2d = torch.randn(25)
a = torch.max(t_2d, [0])
# max_val_2d / max_idxs_2d is of size 1 x 25 -> fine
# The value of max_idxs_2d go from 0 to 4    -> fine


print(t_2d)
print(a)