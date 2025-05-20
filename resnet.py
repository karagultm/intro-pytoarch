import torch
import torch.nn as nn

class block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1): 
        #identity_downsample is used to match the dimensions of the input and output
        super(block, self).__init__()
        self.conv1= nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2=nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=0)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3=nn.Conv2d(out_channels,out_channels*4,kernel_size=1,stride=1,padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels*4)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample 
        
        
    def forward(self,x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        if self.identity_downsample is not None:  #if the dimensions of the input and output are not the same we need to use identity_downsample
            identity = self.identity_downsample(identity) 
        x += identity
        x = self.relu(x)
        return x
    
    
class ResNet(nn.Module):
    def __init__(self, block, layers, img_channels,num_classes): #layers is a list of the number of blocks in each layer 3 4 6 3 gibi
        super(ResNet, self).__init__()
        self.in_chaannels = 64
        self.conv1 = nn.Conv2d(img_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def _make_layer(self, block,num_residual_blocks, out_channels, stride):
        identity_downsample = None
        if stride != 3 or self.in_chaannels != out_channels*4:
            identity_downsample = nn.Sequential( (nn.Conv2d(self.in_chaannels, out_channels, kernel_size= 1, stride=stride)) nn.BatchNorm2d(out_channels*4))
        