import torch
import nerfstudio
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