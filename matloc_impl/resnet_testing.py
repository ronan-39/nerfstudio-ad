import torch
from torchsummary import summary
from urllib.request import urlopen
from PIL import Image
import timm
from torchvision.models import ResNet34_Weights
import torch.nn.init as init
from torch import nn
import copy

class ModifiedResNet(nn.Module):
    def __init__(self):
        super(ModifiedResNet, self).__init__()

        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', weights=ResNet34_Weights.DEFAULT).cuda()

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3

        for param in model.parameters():
            param.requires_grad = False
        
        self.convt1 = nn.ConvTranspose2d(256,256, 3, stride=3).cuda()
        self.convt2 = nn.ConvTranspose2d(256,256, 3, stride=3).cuda()
        self.convt3 = nn.ConvTranspose2d(256,256, 3, stride=2).cuda()
        self.convt4 = nn.ConvTranspose2d(256,256, 3, stride=2).cuda()

        self.adpt_pool = nn.AdaptiveAvgPool2d(output_size=(448,448))

        self.layer4 = model.layer4

        self.cout = nn.Conv2d(512, 64, 1).cuda()
        # self.avgpool = model.avgpool
        # self.fc = nn.Linear(512, 64).cuda()

        

        # for param in model.layer4.parameters():
        #     param.requires_grad = True

        reinit_layer(self.layer4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.convt1(x)
        x = self.convt2(x)
        x = self.convt3(x)
        x = self.convt4(x)

        x = self.adpt_pool(x)

        x = self.layer4(x)

        x = self.cout(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)
        return x

def apply_initializations(layer):
    if isinstance(layer, nn.Conv2d):
        init.xavier_uniform_(layer.weight)  
        if layer.bias is not None:
            init.constant_(layer.bias, 0)
    elif isinstance(layer, nn.BatchNorm2d):
        layer.reset_parameters()
    elif isinstance(layer, nn.Linear):
        layer.reset_parameters()
    

def reinit_layer(layer):
    '''
    recursively reinitialize layers.
    only works on the layer types handled in apply_initializations()
    (Conv2d and BatchNorm2d)
    '''
    has_children = False

    for c in layer.children():
        has_children = True
        reinit_layer(c)

    if not has_children:
        apply_initializations(layer)


# layer4_copy = copy.deepcopy(model.layer4)
# for (p,q) in zip(layer4_copy.parameters(), model.layer4.parameters()):
#     print(torch.allclose(p,q))

def get_modified_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', weights=ResNet34_Weights.DEFAULT).cuda()

    reinit_layer(model.layer4)
    model.fc = nn.Linear(512, 64).cuda()
    apply_initializations(model.fc)

    # for c in model.children():
    #     for param in c.parameters():
    #         param.requires_grad = False

    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer4.parameters():
        param.requires_grad = True
    
    for param in model.fc.parameters():
        param.requires_grad = True

    # for param in model.parameters():
    #     print(param)

    # for name, param in model.named_parameters():
    #     # if param.requires_grad:
    #     print(name, param.data)
    #     print('requires_grad=', param.requires_grad)

    # print(sum(1 for _ in model.parameters()))

    # print(summary(model, (3,224,224)))

    # return model

    return ModifiedResNet()

if __name__ == "__main__":
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', weights=ResNet34_Weights.DEFAULT).cuda()
    # print(model)
    # print(summary(model, (3,224,224)))

    reinit_layer(model.layer4)
    model.fc = nn.Linear(512, 64).cuda()
    apply_initializations(model.fc)

    # print(summary(model, (3,224,224)))

    other_model = ModifiedResNet()
    print(summary(other_model, (3,224,224)))
    # print(model.layer4)