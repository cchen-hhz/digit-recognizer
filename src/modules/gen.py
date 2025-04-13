import numpy as np
import pandas as pd
import os
import yaml
from tqdm import tqdm

import torch
from torch import nn
import torchvision

print('Loading basis...')
with open('args.yaml', 'r') as f:
    params = yaml.safe_load(f)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Preparing datas')
DATA_ROOT = "../Data/digit"
train_url = os.path.join(DATA_ROOT, "train.csv")
test_url = os.path.join(DATA_ROOT, "test.csv")

train_set = pd.read_csv(train_url).values
test_set = pd.read_csv(test_url).values

# 加载数据并进行归一化（将像素值从0-255缩放到0-1）
train_data = torch.tensor(train_set[:, 1:], dtype=torch.float32).view(-1,1,28,28) / 255.0
train_label = torch.tensor(train_set[:, 0], dtype=torch.long)
test_data = torch.tensor(test_set[:, :], dtype=torch.float32).view(-1,1,28,28) / 255.0

def get_data_loader(data, label, batch_size=64, num_workers=6):
    dataset = torch.utils.data.TensorDataset(data, label)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return data_loader

def k_folder(data, label, k, i):
    fold_size = len(data) // k
    train_data = torch.cat((data[:i*fold_size], data[(i+1)*fold_size:]), dim=0)
    train_label = torch.cat((label[:i*fold_size], label[(i+1)*fold_size:]), dim=0)
    val_data = data[i*fold_size:(i+1)*fold_size]
    val_label = label[i*fold_size:(i+1)*fold_size]
    return train_data, train_label, val_data, val_label

# Residual block
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride) if downsample else None
    
    def forward(self, x):
        y = x
        y = self.bn1(self.conv1(y))
        y = self.relu(y)
        y = self.bn2(self.conv2(y))
        if self.downsample is not None:
            x = self.downsample(x)
        y += x
        return self.relu(y)

# ResNet
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage0 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.stage1 = Residual(32, 64, 1, True)
        self.stage2 = Residual(64, 128, 2, True)
        self.stage3 = Residual(128, 128)
        self.final = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.final(x)
        return x

# Train with a specified data
def train_per(train_data, train_label, test_data, test_label):
    train_loader = get_data_loader(train_data, train_label, batch_size=params['batch_size'], num_workers=params['num_workers'])

    train_data_device = train_data.to(device)
    train_label_device = train_label.to(device)
    if test_data is not None:
        test_data_device = test_data.to(device)
        test_label_device = test_label.to(device)

    num_epochs = params['num_epochs']
    net = ResNet().to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params['schedule_step'], gamma=params['schedule_gamma'])

    running_loss = torch.zeros((2, num_epochs)).to(device)
    running_acc = torch.zeros((2, num_epochs)).to(device)

    # 创建进度条并使用自定义格式显示学习率
    pbar = tqdm(range(num_epochs), desc='Training Progress')
    
    # 更新进度条的后缀显示格式以包含学习率
    pbar.bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    
    for i in pbar:
        # 更新进度条后缀显示当前学习率
        pbar.set_postfix_str(f"lr: {optimizer.param_groups[0]['lr']:.1e}")
        
        net.train()
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = net(x)
            loss_value = loss(pred, y)
            loss_value.backward()
            optimizer.step()
        
        # 学习率调度器步进
        scheduler.step()
        
        net.eval()
        with torch.no_grad():
            # Compute train_set loss
            pred = net(train_data_device)
            loss_value = loss(pred, train_label_device)
            running_loss[0, i] = loss_value.item()
            running_acc[0, i] = (pred.argmax(dim=1) == train_label_device).float().mean().item()

            if test_data_device is not None:
                # Compute test_set loss
                pred = net(test_data_device)
                loss_value = loss(pred, test_label_device)
                running_loss[1, i] = loss_value.item()
                running_acc[1, i] = (pred.argmax(dim=1) == test_label_device).float().mean().item()

    running_loss = running_loss.cpu()
    running_acc = running_acc.cpu()
    return net, running_loss, running_acc

# 添加用于可视化训练过程的Animator类
class Animator:
    def __init__(self, xlabel='Epochs', figsize=(15, 6)):
        """初始化动画器，创建两个子图分别显示损失和准确率
        
        Parameters:
        -----------
        xlabel : str
            x轴标签
        figsize : tuple
            图表尺寸
        """
        # 导入matplotlib，放在类内部以避免不必要的全局导入
        import matplotlib.pyplot as plt
        self.plt = plt
        
        # 创建两个子图
        self.fig, (self.ax_loss, self.ax_acc) = plt.subplots(1, 2, figsize=figsize)
        
        # 设置标题和标签
        self.ax_loss.set_title("Training and Validation Loss")
        self.ax_loss.set_xlabel(xlabel)
        self.ax_loss.set_ylabel("Loss")
        
        self.ax_acc.set_title("Training and Validation Accuracy")
        self.ax_acc.set_xlabel(xlabel)
        self.ax_acc.set_ylabel("Accuracy")
        
        # 初始化数据存储
        self.X = []
        self.Y_loss = [[], []]  # [train_loss, val_loss]
        self.Y_acc = [[], []]   # [train_acc, val_acc]
        
        # 设置线条颜色和样式
        self.colors = ['#1f77b4', '#ff7f0e']
        self.loss_lines = []
        self.acc_lines = []
        
        # 初始化线条对象
        for i, label in enumerate(['Training Loss', 'Validation Loss']):
            line, = self.ax_loss.plot([], [], '-', color=self.colors[i], label=label)
            self.loss_lines.append(line)
            
        for i, label in enumerate(['Training Accuracy', 'Validation Accuracy']):
            line, = self.ax_acc.plot([], [], '-', color=self.colors[i], label=label)
            self.acc_lines.append(line)
        
        # 显示图例
        self.ax_loss.legend()
        self.ax_acc.legend()
        
        # 初始化网格
        self.ax_loss.grid(True)
        self.ax_acc.grid(True)
        
        # 用于自动调整y轴范围
        self.y_min_loss = float('inf')
        self.y_max_loss = float('-inf')
        self.y_min_acc = float('inf') 
        self.y_max_acc = float('-inf')
    
    def add(self, x, y):
        """添加一个新的数据点
        
        Parameters:
        -----------
        x : number or list
            x轴数据点或数据点列表
        y : list or numpy.ndarray
            形状为(4, len(x))的数组，包含四条曲线的y值
            [train_loss, val_loss, train_acc, val_acc]
        """
        import numpy as np
        
        # 确保x是列表
        if not isinstance(x, list) and not isinstance(x, np.ndarray):
            x = [x]
        
        # 确保y是numpy数组
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        
        # 添加数据
        self.X.extend(x)
        
        # 分别添加损失和准确率数据
        for i in range(2):
            self.Y_loss[i].extend(y[i])
            self.Y_acc[i].extend(y[i+2])
            
        # 更新损失图表的y轴范围
        y_min_loss = min([min(series) if series else float('inf') for series in self.Y_loss])
        y_max_loss = max([max(series) if series else float('-inf') for series in self.Y_loss])
        
        self.y_min_loss = min(self.y_min_loss, y_min_loss)
        self.y_max_loss = max(self.y_max_loss, y_max_loss)
        
        # 为损失图表添加上下边距
        margin_loss = (self.y_max_loss - self.y_min_loss) * 0.1 if self.y_max_loss > self.y_min_loss else 0.1
        self.ax_loss.set_ylim(max(0, self.y_min_loss - margin_loss), self.y_max_loss + margin_loss)
        
        # 更新准确率图表的y轴范围
        y_min_acc = min([min(series) if series else float('inf') for series in self.Y_acc])
        y_max_acc = max([max(series) if series else float('-inf') for series in self.Y_acc])
        
        self.y_min_acc = min(self.y_min_acc, y_min_acc)
        self.y_max_acc = max(self.y_max_acc, y_max_acc)
        
        # 为准确率图表添加上下边距
        margin_acc = (self.y_max_acc - self.y_min_acc) * 0.1 if self.y_max_acc > self.y_min_acc else 0.1
        self.ax_acc.set_ylim(max(0, self.y_min_acc - margin_acc), min(1.0, self.y_max_acc + margin_acc))
        
        # 更新x轴范围
        x_min = min(self.X)
        x_max = max(self.X)
        self.ax_loss.set_xlim(x_min, x_max)
        self.ax_acc.set_xlim(x_min, x_max)
        
        # 更新线条数据
        for i, line in enumerate(self.loss_lines):
            line.set_data(self.X, self.Y_loss[i])
            
        for i, line in enumerate(self.acc_lines):
            line.set_data(self.X, self.Y_acc[i])
        
    def show(self):
        """显示当前图表"""
        # 设置适当的边距
        self.fig.tight_layout()
        
        # 显示图表
        self.plt.show()
        
    def save(self, filename):
        """保存图表到文件
        
        Parameters:
        -----------
        filename : str
            保存的文件名
        """
        self.fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"图表已保存到 {filename}")

def train_with_k_folder(k):
    animator = Animator()
    num_epochs = params['num_epochs']
    loss_total = torch.zeros((2, params['num_epochs']))
    acc_total = torch.zeros((2, params['num_epochs']))
    
    for i in range(k):
        print(f'Fold {i+1}/{k}')
        train_dataset, train_labelset, val_dataset, val_labelset = k_folder(train_data, train_label, k, i)
        net, loss, acc = train_per(train_dataset, train_labelset, val_dataset, val_labelset)
        
        loss_total[0] += loss[0]
        loss_total[1] += loss[1]
        acc_total[0] += acc[0]
        acc_total[1] += acc[1]
    
    print('Computing')
    loss_total /= k
    acc_total /= k
    animator.add(np.arange(num_epochs), 
    [loss_total[0].numpy(), loss_total[1].numpy(),
    acc_total[0].numpy(), acc_total[1].numpy()])
    animator.show()
    #animator.save('train.png')
    #print('Finished, saved as train.png')

def predict(net, data):
    net.eval()
    data = data.to(device)
    with torch.no_grad():
        pred = net(data)
        pred = pred.argmax(dim=1).cpu().numpy()
    return pred


print('params detail-----')
for name, val in params.items():
    print(f'{name}:{val}')
print('------')
print(f'device on {device}')
print('Start training\n')

#train_with_k_folder(params['k_fold'])

net, _, _ = train_per(train_data, train_label, None, None)

# Saving
torch.save(net.state_dict(), 'resnet.pth')




