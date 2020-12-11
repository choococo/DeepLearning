import torch
import torch.nn as nn

a = torch.randn(100, 3, 10, 224, 224)  # [N, C, D, H, W]
conv_3d = nn.Conv3d(3, 16, kernel_size=(2, 3, 3), stride=2, padding=1)

b = conv_3d(a)
print(b.shape)