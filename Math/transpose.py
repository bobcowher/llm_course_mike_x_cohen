import numpy as np
import torch

# ones = np.ones(10)

nv = np.array([[1,2,3,4]])

print(nv)

# nv_t = nv.transpose()
nv_t = nv.T

print(nv_t)


nM = np.array(
    [
        [1,2,3,4],
        [5,6,7,8]
    ]
)
print("-----")

print(nM)

print(nM.T)


print("\nPyTorch")

tv = torch.tensor([[1,2,3,4]])
print(tv)

tv_t = tv.T

print(tv_t)

