import sys
import numpy as np
import os
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms, models
import torchvision.models as models
from PIL import Image


os.environ['GLOG_v'] = '2'
has_gpu = torch.cuda.is_available()
print('Executing with', 'GPU' if has_gpu else 'CPU', '.')
device = torch.device('cuda' if has_gpu else 'cpu')

index = {'00_00': 0, '00_01': 1, '00_02': 2, '00_03': 3, '00_04': 4, '00_05': 5, '00_06': 6, '00_07': 7,
         '00_08': 8, '00_09': 9, '01_00': 10, '01_01': 11, '01_02': 12, '01_03': 13, '01_04': 14,
         '01_05': 15, '01_06': 16, '01_07': 17, '02_00': 18, '02_01': 19, '02_02': 20, '02_03': 21,
         '03_00': 22, '03_01': 23, '03_02': 24, '03_03': 25}
inverted = {0: 'Plastic Bottle', 1: 'Hats', 2: 'Newspaper', 3: 'Cans', 4: 'Glassware', 5: 'Glass Bottle', 6: 'Cardboard', 7: 'Basketball',
            8: 'Paper', 9: 'Metalware', 10: 'Disposable Chopsticks', 11: 'Lighter', 12: 'Broom', 13: 'Old Mirror', 14: 'Toothbrush',
            15: 'Dirty Cloth', 16: 'Seashell', 17: 'Ceramic Bowl', 18: 'Paint bucket', 19: 'Battery', 20: 'Fluorescent lamp', 21: 'Tablet capsules',
            22: 'Orange Peel', 23: 'Vegetable Leaf', 24: 'Eggshell', 25: 'Banana Peel'}


# 训练超参
config = {
    "num_classes": 26,
    "image_height": 224,
    "image_width": 224,
    "batch_size": 32,
    "eval_batch_size": 10,
    "epochs": 50,
    "lr_max": 0.001,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "dataset_path": "./datasets/5fbdf571c06d3433df85ac65-momodel/garbage_26x100",
    "features_path": "./results/garbage_26x100_features",
    "class_index": index,
    "save_ckpt_epochs": 1,
    "save_ckpt_path": './results/ckpt_efficientnet',
    "export_path": './results/efficientnet.pth'
}

transform = transforms.Compose([
    transforms.ToPILImage(),  # 将 numpy array 转换为 PIL Image
    transforms.Resize((config['image_height'], config['image_width'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=26, pretrained=True, pretrained_path=None):
        super(EfficientNetClassifier, self).__init__()
        # 使用EfficientNet-B0作为backbone
        self.backbone = models.efficientnet_b0(pretrained=False)
        
        # 如果提供了本地预训练权重路径,则加载
        if pretrained and pretrained_path:
            if os.path.exists(pretrained_path):
                print(f"Loading pretrained weights from {pretrained_path}")
                state_dict = torch.load(pretrained_path, map_location='cpu')
                # 处理可能的权重键名不匹配问题
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                # 只加载backbone的权重,忽略分类头
                try:
                    self.backbone.load_state_dict(state_dict, strict=False)
                    print("Pretrained weights loaded successfully!")
                except Exception as e:
                    print(f"Warning: Could not load some weights: {e}")
                    print("Continuing with partially loaded weights...")
            else:
                print(f"Warning: Pretrained weights file not found at {pretrained_path}")
                print("Continuing with random initialization...")
        
        # 获取原始分类器的输入特征数
        in_features = self.backbone.classifier[1].in_features
        
        # 替换分类头
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
    
    def extract_features(self, x):
        # 提取特征(不包括分类头)
        return self.backbone.features(x)
    

pretrained_weights_path = "./results/best_model.pth"
model = EfficientNetClassifier(
    num_classes=config['num_classes'], 
    pretrained=False,  # 不加载预训练权重
)
# 直接加载完整的训练好的模型权重
checkpoint = torch.load(pretrained_weights_path, map_location='cpu')
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()  # 设置为评估模式


def predict(image):
    """
    加载模型和模型预测
    主要步骤:
        1.图片处理,此处尽量与训练模型数据处理一致
        2.用加载的模型预测图片的类别
    :param image: PIL 读取的图片对象，数据类型是 np.array，shape (H, W, C)
    :return: string, 模型识别图片的类别,
            包含 'Plastic Bottle','Hats','Newspaper','Cans'等共 26 个类别
    """
    # -------------------------- 实现图像处理部分的代码 ---------------------------
    image = transform(image).to(device)
    image = torch.unsqueeze(image, dim=0)

    # -------------------------- 实现模型预测部分的代码 ---------------------------
    with torch.no_grad():  # 推理时不需要梯度
        logits = model(image)
        pred = torch.argmax(logits, dim=1).cpu().numpy()[0]

    return inverted[pred]
    