import torch
from torchvision import transforms

'设备'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'图像数据归一化处理'
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 对于图像这个是有经验值的
])

'回归框'
BOUND_BOX = [12, 24, 48]
