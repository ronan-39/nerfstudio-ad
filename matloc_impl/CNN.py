import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet18Pretrained(nn.Module):
    def __init__(self):
        super(ResNet18Pretrained, self).__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
        modules = list(self.resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        for p in self.resnet.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.resnet(x)

class ResNet152Pretrained(nn.Module):
    def __init__(self):
        super(ResNet152Pretrained, self).__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet152', pretrained=True)
        modules = list(self.resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        for p in self.resnet.parameters():
            p.requires_grad = False
            
    def forward(self, x):
        return self.resnet(x)
    
class ResNet34Pretained(nn.Module):
    def __init__(self):
        super(ResNet34Pretained, self).__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet34', pretrained=True)
        modules = list(self.resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        for p in self.resnet.parameters():
            p.requires_grad = False
            
    def forward(self, x):
        return self.resnet(x)