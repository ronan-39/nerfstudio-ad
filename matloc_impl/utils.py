import torch
from torch import Tensor
from torchvision.transforms import ToPILImage
import nerfstudio
from nerfstudio.utils.colormaps import ColormapOptions
import numpy as np
from nerfstudio.cameras.cameras import Cameras

def show_mem_usage():
    usage = torch.cuda.mem_get_info()
    print("mem usage: ", 1 - usage[0]/usage[1])

def gen_camera():

    fov = np.pi/2.0
    image_width = 256 * 2 # in pixels
    image_height = 256 * 2
    pp_w = image_width / 2.0
    pp_h = image_height / 2.0
    focal_length = pp_h / np.tan(fov / 2.0)
    intrinsics_matrix = torch.tensor([[focal_length, 0, pp_w], [0, focal_length, pp_h], [0, 0, 1]], dtype=torch.float32)
    fx = intrinsics_matrix[0, 0]
    fy = intrinsics_matrix[1, 1]

    transform_matrix = [
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0.5]
    ]

    return Cameras(torch.tensor(transform_matrix), fx, fy, pp_w, pp_h)


def display_depth_image(o, filter=False): # input a depth tensor TODO: assert that its the correct input

    o = o.permute(2,0,1)

    if filter:
    # make the depth image look better (not sure how this affects other outputs)
        _max = o.max()
        _min = o.min()

        o -= _min.item()
        o /= _max.item()
        o = np.power(o, 1/4) # tune the exponent per NeRF, 1/4 is good for the gorilla
        o *= 255.0
        o = np.where( o > 80.0, 255.0, o)

    to_pil = ToPILImage()
    image = to_pil(o) # use this if you use `o.where()` (it becomes an array instead of tensor)
    # image = to_pil(o.reshape([1,256,256]).byte()) 

    image.show()

def display_features_image(o):
    """Map a high dimension feature vector to an RGB color space and display it
    """

    a = nerfstudio.utils.colormaps.apply_colormap(o, colormap_options=ColormapOptions())
    a = a.permute(2,0,1)

    to_pil = ToPILImage()
    image = to_pil(a)

    image.show()