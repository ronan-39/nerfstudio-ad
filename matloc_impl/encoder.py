"""
A CNN to encode images to the space of NeRF features
"""

import torch
import torchvision
import torchvision.models as models
from torch import Tensor, nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torch.nn.functional as F
import torch.optim
from torchvision import datasets,transforms
import numpy as np
from pathlib import PosixPath, Path
import os
import pickle
import matplotlib.pyplot as plt
import nerfstudio
from nerfstudio.utils.colormaps import ColormapOptions

# load resnet 18
# model = models.resnet18(weights=None)

# remove the last layer
# model = nn.Sequential(*(list(model.children())[:-1]))

# print(model)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.en_conv1 = nn.Conv2d(3, 16, 3, padding=1, stride=1)
        self.en_conv2 = nn.Conv2d(16, 32, 3, padding=1, stride=1)
        self.en_conv3 = nn.Conv2d(32, 64, 3, padding=1, stride=1)    

    def forward(self, x):
        x = F.relu(self.en_conv1(x))
        x = F.relu(self.en_conv2(x))
        x = self.en_conv3(x)
        return x

class EncodingDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__()

        self.data_dir = data_dir
        self.len = self.__len__()

    def __len__(self):
        return len([name for name in os.listdir(self.data_dir.joinpath("features")) if name.endswith(".feature")])


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if idx > self.len:
            raise Exception("Out of bounds")

        input_filepath = self.data_dir.joinpath(f'rgb/rgb_{idx}.png')
        output_filepath = self.data_dir.joinpath(f'features/feat_{idx}.feature')

        input = torchvision.io.read_image(str(input_filepath))

        output_file = open(output_filepath, 'rb')
        output = pickle.load(output_file)

        if output.shape[2] == 64: # its in the wrong order, a remnant of earlier code
            output = output.permute(2,0,1)
            
        return {'input': input, 'desired_output': output}
            
def preview_dataset(dataset):

    fig, axes = plt.subplots(nrows=2, ncols=6, sharex=True, sharey=True, figsize=(48,14))

    ins = [dataset.__getitem__(i)['input'].permute(1,2,0).numpy() for i in range(6)]
    outs = [nerfstudio.utils.colormaps.apply_colormap(dataset.__getitem__(i)['desired_output'].cpu().permute(1,2,0), colormap_options=ColormapOptions()).numpy() for i in range(6)]

    for imgs,row in zip([ins, outs], axes):
        for img, ax in zip(imgs, row):
            ax.imshow(np.squeeze(img), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    plt.show()

if __name__ == "__main__":
    print("running encoder.py")
    model = Encoder()

    if torch.cuda.is_available():
        model.cuda()

    training_data_dir = Path("./outputs/unnamed/feature_image_pairs")

    if not training_data_dir.exists():
        print("The training data directory does not exist. Exiting.")
        import sys
        sys.exit()

    encoder_model_dir = training_data_dir.joinpath("encoder_models")
    if not Path.exists(encoder_model_dir):
        Path.mkdir(encoder_model_dir)
      
    valid_size = 0.2
    batch_size = 20
    num_workers=8

    dataset = EncodingDataset(training_data_dir)
    
    # preview_dataset(dataset)

    dataloader = DataLoader(dataset, batch_size=5, sampler=SubsetRandomSampler(list(range(dataset.len))), num_workers=0)
    

"""
ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (1): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer2): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer3): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer4): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): Linear(in_features=512, out_features=1000, bias=True)
)
"""