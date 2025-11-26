import math
import sys
import numpy as np
import os
import cv2
import random
import shutil
import time
from matplotlib import pyplot as plt
import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torchvision.models as models

os.environ['GLOG_v'] = '2'
has_gpu = torch.cuda.is_available()
print('Executing with', 'GPU' if has_gpu else 'CPU', '.')
device = torch.device('cuda:0' if has_gpu else 'cpu')

# 垃圾分类数据集标签,以及用于标签映射的字典。
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
    "save_ckpt_path": './results/ckpt_efficientnet2',
    "export_path": './results/efficientnet.pth'
}


class GarbageDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.images = []
        self.labels = []
        
        if not os.path.exists(data_path):
            raise ValueError(f"Dataset path not found: {data_path}")

        # 遍历数据集目录
        for class_name in os.listdir(data_path):
            class_path = os.path.join(data_path, class_name)
            if os.path.isdir(class_path) and class_name in index:
                label = index[class_name]
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(img_path)
                        self.labels.append(label)
        
        if len(self.images) == 0:
            print(f"Error: No images found in {data_path}")
            print(f"Expected class names (first 5): {list(index.keys())[:5]}...")
            if os.path.exists(data_path):
                print(f"Actual folders found (first 5): {os.listdir(data_path)[:5]}...")
            raise ValueError("Dataset is empty. Please check if the dataset path is correct and folder names match the index keys.")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def create_dataloader(config, is_train=True):
    ds = ""
    if is_train:
        transform = transforms.Compose([
            transforms.Resize((config['image_height'], config['image_width'])),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        batch_size = config['batch_size']
        shuffle = True
        ds = os.path.join(config['dataset_path'], 'train')
    else:
        transform = transforms.Compose([
            transforms.Resize((config['image_height'], config['image_width'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        batch_size = config['eval_batch_size']
        shuffle = False
        ds = os.path.join(config['dataset_path'], 'val')
    
    dataset = GarbageDataset(ds, transform=transform)
            

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1)
    
    return dataloader


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


def build_lr_scheduler(optimizer, total_steps, warmup_steps=0):
    """构建学习率调度器"""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_model():
    # 创建数据加载器
    train_loader = create_dataloader(config, is_train=True)
    eval_loader = create_dataloader(config, is_train=False)

    
    # 创建模型,使用本地预训练权重
    pretrained_weights_path = "./src/efficientnet_b0_rwightman-7f5810bc.pth"
    model = EfficientNetClassifier(
        num_classes=config['num_classes'], 
        pretrained=True,
        pretrained_path=pretrained_weights_path
    )
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config['lr_max'], 
                         momentum=config['momentum'], weight_decay=config['weight_decay'])
    
    # 学习率调度器
    total_steps = config['epochs'] * len(train_loader)
    scheduler = build_lr_scheduler(optimizer, total_steps, warmup_steps=0)
    
    # 训练循环
    history = []
    best_acc = 0.0
    
    for epoch in tqdm.tqdm(range(config['epochs'])):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_seconds = time.time() - epoch_start
        history.append(epoch_loss)
        
        print(f"Epoch: {epoch + 1}/{config['epochs']}, Time: {epoch_seconds:.2f}s, Avg Loss: {epoch_loss:.4f}")
        
        # 验证
        if (epoch + 1) % config['save_ckpt_epochs'] == 0:
            model.eval()
            correct = 0
            total = 0
            eval_loss = 0.0
            
            with torch.no_grad():
                for images, labels in eval_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    eval_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            accuracy = 100 * correct / total
            eval_loss = eval_loss / len(eval_loader)
            print(f"Validation - Loss: {eval_loss:.4f}, Accuracy: {accuracy:.2f}%")
            
            # 保存最佳模型
            if accuracy > best_acc:
                best_acc = accuracy
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': accuracy,
                }, os.path.join(config['save_ckpt_path'], f'best_model.pth'))
    
    print(f"\nTraining completed! Best accuracy: {best_acc:.2f}%")
    return history, model


if __name__ == '__main__':
    if os.path.exists(config['save_ckpt_path']):
        shutil.rmtree(config['save_ckpt_path'])
    os.makedirs(config['save_ckpt_path'])
    
    history, model = train_model()
    
    print('Training is complete!')
    
    # 保存最终模型
    torch.save(model.state_dict(), config['export_path'])
    print(f"Final model saved to {config['export_path']}")