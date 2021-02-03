import torch
from torchvision import models

output = torch.randn(1, 13, 13, 3, 15)
target = torch.randn(1, 13, 13, 3, 15)
mask_obj = target[..., 0] > 0
# print(mask_obj.shape)
# output_obj = output[mask_obj]
# print(output_obj.shape)
# idxs = mask_obj.nonzero(as_tuple=False)
# vecs = output[mask_obj]
# print(idxs)
# print(vecs.shape)
#
# a = idxs[:, 3]
# print(a)
#
torch.manual_seed(0)
a = torch.randn(2, 3, 3, 3)
a = torch.Tensor([
    [
        [
            [1, 2, 3],
            [3, 2, 1]
        ],
        [
            [2, 2, 3],
            [3, 2, 2]
        ],
        [
            [3, 2, 3],
            [3, 2, 3]
        ]
    ]
])
print(a)
nu2 = torch.pow(a, 2).mean(dim=[2, 3], keepdim=True)
print(nu2)
