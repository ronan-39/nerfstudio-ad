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
    print(type(cam))

    # tell the model's field to output the activations of a "intermediate"(hidden) layer
    pipeline.model.field.add_intermediate_outputs([1])

    # get outputs of the NeRF from the camera's POV
    outputs = pipeline.model.get_outputs_for_camera(cam)

    print(outputs.keys())

    utils.display_features_image(outputs['layer1'])
    utils.display_depth_image(outputs['depth'], filter=False)

    # print(type(MODEL_PATH))

    # training_data = utils.CNNTrainingData("./01Gorilla/transforms.json")
    # print(training_data)

def train():
    #assign device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #parse the arguments from config
    parser = config_parser()
    args = parser.parse_args()
    output_dir = args.output_dir
    model_name = args.model_name
    batch_size = args.batch_size
    kernel_size = args.kernel_size
    lrate = args.lrate
    MODEL_PATH = Path(args.model_path)
    sampling_strategy = args.sampling_strategy
    print(MODEL_PATH)
    #Load PAD classes
    class_name = 'all'
    class_names = lego.CLASS_NAMES if class_name == 'all' else [
        class_name]

    #load the pretrained NERF model
    config, pipeline, _, _ = eval_setup(
            config_path=MODEL_PATH,
            test_mode='inference'
        )
    
    # tell the model's field to output the activations of a "intermediate"(hidden) layer
    pipeline.model.field.add_intermediate_outputs([1])
    
    #iterate over all classes, and then all images in the class
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
            
            #remove last row of start pose for ease of use with nerfstudio
            start_pose = start_pose[:-1]
            
            #convert to torch tensor
            start_pose = torch.tensor(start_pose).to(device) 
            
            cam = utils.gen_camera(transform_matrix = start_pose)
            
            # get outputs of the NeRF from the camera's POV
            outputs = pipeline.model.get_outputs_for_camera(cam)
                        
            #get features and depth images
            features = outputs['layer1']
            depth = outputs['depth']
            
            print(features.shape)
            print(depth.shape)

            
            
            
            
            
            
            




if __name__ == '__main__':
    # main()
    train()
