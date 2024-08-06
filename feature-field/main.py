import nerfstudio
from nerfstudio.field_components.encodings import HashEncoding, SHEncoding
import torch
from torch import nn
from nerfstudio.field_components.mlp import MLP, MLPWithHashEncoding
from torchsummary import summary

'''
The Feature Field will encode the features corresponding to a view of an object from a given angle.
The object is assumed to be at the origin of the feature field.

The output feature should be trained to match some known feature from ResNet or such.
'''

class FF(nn.Module):
    def __init__(self):
        super(FF, self).__init__()

        self.view_encoder = SHEncoding()#(implementation="tcnn")
        self.mlp_base = MLP(
            num_layers=6,
            layer_width=128,
            in_dim = self.view_encoder.get_out_dim(),
            out_dim = 64,
            activation=nn.ReLU(),
            out_activation=None,
            implementation="tcnn",
        )

    def forward(self, x):
        x = x.view(-1, 3)
        x = self.view_encoder(x)
        x = self.mlp_base(x)
        
        return x
        



ff = FF().cuda()
print(summary(ff, (1,3)))
# print(ff.view_encoder)
