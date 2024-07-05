import nerfstudio
import torch
import numpy as np
import cv2
import os
from nerfstudio.models.base_model import Model
from nerfstudio.cameras.rays import Frustums, RaySamples
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.models import nerfacto
from pathlib import Path
from nerfstudio.utils.eval_utils import eval_setup
from torchvision.transforms import ToPILImage
from torch import Tensor, nn
from typing import Generator, Literal, Optional, Tuple, Union
from jaxtyping import Float, Int
from nerfstudio.field_components.field_heads import FieldHeadNames
from collections import defaultdict
from utils import (config_parser,
                   load_blender_ad,pose_retrieval_loftr)
import datasets.LEGO_3D as lego
from datasets.LEGO_3D import LEGODataset
from torch.utils.data import DataLoader




import utils

def main():
    MODEL_PATH = Path('outputs/unnamed/nerfacto/2024-06-27_170932/config.yml')
    assert Path('outputs/unnamed/nerfacto/2024-06-27_170932/nerfstudio_models/step-000029999.ckpt').exists(), "The checkpoint file wasn't found."
    
    config, pipeline, _, _ = eval_setup(
        config_path=MODEL_PATH,
        test_mode='inference'
    )

    # generate a camera with simple params pointed at the center of the scene
    cam = utils.gen_camera()

    # tell the model's field to output the activations of a "intermediate"(hidden) layer
    pipeline.model.field.add_intermediate_outputs([1])

    # get outputs of the NeRF from the camera's POV
    outputs = pipeline.model.get_outputs_for_camera(cam)

    print(outputs.keys())

    utils.display_features_image(outputs['layer1'])
    utils.display_depth_image(outputs['depth'], filter=False)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    parser = config_parser()
    args = parser.parse_args()
    output_dir = args.output_dir
    model_name = args.model_name
    batch_size = args.batch_size
    kernel_size = args.kernel_size
    lrate = args.lrate
    sampling_strategy = args.sampling_strategy
    class_name = 'all'

    class_names = lego.CLASS_NAMES if class_name == 'all' else [
        class_name]

    for class_name in class_names:
        # load the good imgs with their poses
        imgs, hwf, poses = load_blender_ad(
            args.data_dir, model_name, args.half_res, args.white_bkgd)
        H, W, focal = hwf
        near, far = 2., 6.  # Blender

        # load the anomaly image
        lego_dataset = LEGODataset(dataset_path=args.data_dir,
                                   class_name=class_name,
                                   resize=400)

        lego_loader = DataLoader(dataset=lego_dataset,
                                 batch_size=1,
                                 pin_memory=False)
        test_imgs = list()
        gt_mask_list = list()
        gt_list = list()
        
        index = 0
        for x, y, mask in lego_loader:
            test_imgs.extend(x.cpu().numpy())
            gt_list.extend(y.cpu().numpy())
            mask = (mask.cpu().numpy()/255.0).astype(np.uint8)
            gt_mask_list.extend(mask)

            obs_img = x.cpu().numpy().squeeze(axis=0)
            # Find the start pose by looking for the most similar images
            start_pose = pose_retrieval_loftr(imgs, obs_img, poses)
            print(f"Start pose: {start_pose}")
            print("Type of start pose: ", type(start_pose))
            print(start_pose.shape)


            # Create pose transformation model
            start_pose = torch.Tensor(start_pose).to(device)




if __name__ == '__main__':
    train()
