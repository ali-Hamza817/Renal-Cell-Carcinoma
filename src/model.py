import torch
import torch.nn as nn
import torch.nn.functional as F

class Tiny3DCNN(nn.Module):
    """
    A lightweight 3D CNN for quick testing, debugging, and baseline verification.
    Input Shape: (B, 1, 64, 128, 128)
    """
    def __init__(self):
        super(Tiny3DCNN, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(8)
        self.pool1 = nn.MaxPool3d(2)  # Output: 32 x 64 x 64
        
        # Block 2
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(16)
        self.pool2 = nn.MaxPool3d(2)  # Output: 16 x 32 x 32
        
        # Block 3
        self.conv3 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(32)
        self.pool3 = nn.MaxPool3d(2)  # Output: 8 x 16 x 16
        
        # Block 4
        self.conv4 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(64)
        self.pool4 = nn.MaxPool3d(2)  # Output: 4 x 8 x 8
        
        # Pooling & Classification
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))  # Output: (B, 64, 1, 1, 1)
        self.fc1 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 1)  # Outputs raw logit
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class BasicBlock3D(nn.Module):
    """
    3D Basic Residual Block (Conv3D -> BN3D -> ReLU -> Conv3D -> BN3D + Shortcut)
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet3D(nn.Module):
    """
    General 3D ResNet class that can instantiate ResNet-10, ResNet-18, etc.
    Input Shape: (B, 1, 64, 128, 128)
    """
    def __init__(self, block, num_blocks, num_classes=1):
        super(ResNet3D, self).__init__()
        self.in_planes = 16

        # Stem: Initial Conv layer
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(16)
        
        # Layer blocks
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)   # (B, 16, 64, 128, 128)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)   # (B, 32, 32, 64, 64)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)   # (B, 64, 16, 32, 32)
        self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=2)  # (B, 128, 8, 16, 16)
        
        # Pooling & Classification
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(128 * block.expansion, num_classes)
        self.dropout = nn.Dropout(0.4)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

def get_resnet3d_10():
    """Returns a 3D ResNet-10 model."""
    return ResNet3D(BasicBlock3D, [1, 1, 1, 1])

def get_resnet3d_18():
    """Returns a 3D ResNet-18 model."""
    return ResNet3D(BasicBlock3D, [2, 2, 2, 2])

if __name__ == "__main__":
    # Test model shape validation
    x = torch.randn(2, 1, 64, 128, 128)
    
    tiny_cnn = Tiny3DCNN()
    out_cnn = tiny_cnn(x)
    print(f"Tiny3DCNN input shape: {x.shape} -> output shape: {out_cnn.shape}")
    
    resnet18 = get_resnet3d_18()
    out_res18 = resnet18(x)
    print(f"ResNet3D-18 input shape: {x.shape} -> output shape: {out_res18.shape}")
