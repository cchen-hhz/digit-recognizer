from flask import Flask, request, jsonify
import torch
import numpy as np
from torch import nn
from flask_cors import CORS

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
# 初始化 Flask 应用
app = Flask(__name__)

# 启用 CORS
CORS(app)

# 修改模型加载和推理代码，将模型和数据都放在 CPU 上
# 加载模型
model = ResNet().to('cpu')  # 强制将模型加载到 CPU
model.load_state_dict(torch.load('resnet.pth', map_location='cpu'))  # 确保模型权重加载到 CPU
model.eval()

# 定义预测 API
@app.route('/predict', methods=['POST'])
def predict():
    print("Received request")
    data = request.json.get('image')  # 获取前端传来的图像数据
    if not data:
        return jsonify({'error': 'No image data provided'}), 400

    # 将图像数据转换为张量
    image = torch.tensor(data, dtype=torch.float32).view(1, 1, 28, 28).to('cpu')  # 确保数据在 CPU 上
    with torch.no_grad():
        output = model(image)  # 在 CPU 上进行推理
        prediction = output.argmax(dim=1).item()

    print(f"Prediction: {prediction}")  # 打印预测结果
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)