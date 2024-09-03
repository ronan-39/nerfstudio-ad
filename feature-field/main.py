from nerfstudio.field_components.encodings import HashEncoding, SHEncoding
from nerfstudio.field_components.mlp import MLP, MLPWithHashEncoding
from torchsummary import summary
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
                   load_blender_ad,pose_retrieval_loftr, find_POI)
from models.feature_extractors import (get_rays, cost_based_matching)
import datasets.LEGO_3D as lego
from datasets.LEGO_3D import LEGODataset
from torch.utils.data import DataLoader
from models.NFF import (load_NFF,generate_features)
import warnings

'''
The Feature Field will encode the features corresponding to a view of an object from a given angle.
The object is assumed to be at the origin of the feature field.

The output feature should be trained to match some known feature from ResNet or such.
'''


#training code for the feature field: use the pad 
def train_feature_field():
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
    #Load PAD classes
    class_name = 'all'
    class_names = lego.CLASS_NAMES if class_name == 'all' else [
        class_name]
    
     #model loading
    render_kwargs, feature_field = load_NFF(args, device)
    
    
    optimizer = torch.optim.Adam(
                params=feature_field.parameters(), lr=lrate, betas=(0.9, 0.999))
    #iterate over all classes, and then all images in the class
    for class_name in class_names:
        
        # load the good imgs with their poses
        imgs, hwf, poses = load_blender_ad(
            args.data_dir, model_name, args.half_res, args.white_bkgd)
        H, W, focal = hwf
        near, far = 2., 6.  # Blender
      
        # Move testing data to GPU
        poses = torch.Tensor(poses).to(device)
        



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
    #Load PAD classes
    class_name = 'all'
    class_names = lego.CLASS_NAMES if class_name == 'all' else [
        class_name]

    # #load the pretrained NERF model
    # config, pipeline, _, _ = eval_setup(
    #         config_path=MODEL_PATH,
    #         test_mode='inference'
    #     )
    
    # # tell the model's field to output the activations of a "intermediate"(hidden) layer
    # pipeline.model.field.add_intermediate_outputs([1])
    
    #load dinov2 for dense features, note, requires patches of size 14
    dinov2_vits14_lc = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_lc').to(device)
    
    
    #model loading
    render_kwargs, feature_field = load_NFF(args, device)
    
    
    optimizer = torch.optim.Adam(
                params=feature_field.parameters(), lr=lrate, betas=(0.9, 0.999))
    #iterate over all classes, and then all images in the class
    for class_name in class_names:
        
        # load the good imgs with their poses
        imgs, hwf, poses = load_blender_ad(
            args.data_dir, model_name, args.half_res, args.white_bkgd)
        H, W, focal = hwf
        near, far = 2., 6.  # Blender

        # load the anomaly image, note needed a downsize to 224x224
        lego_dataset = LEGODataset(dataset_path=args.data_dir,
                                   class_name=class_name,
                                   resize=224)

        lego_loader = DataLoader(dataset=lego_dataset,
                                 batch_size=1,
                                 pin_memory=False)
        test_imgs = list()
        gt_mask_list = list()
        gt_list = list()
        
        index = 0
        
        for epoch in range(args.epochs):

            for x, y, mask in lego_loader:
                optimizer.zero_grad()
                
                test_imgs.extend(x.cpu().numpy())
                gt_list.extend(y.cpu().numpy())
                mask = (mask.cpu().numpy()/255.0).astype(np.uint8)
                gt_mask_list.extend(mask)

                obs_img = x.cpu().numpy().squeeze(axis=0)
                
                # Find the start pose by looking for the most similar images
                start_pose = pose_retrieval_loftr(imgs, obs_img, poses)
                
                POI = find_POI(obs_img, False)
                obs_img = (np.array(obs_img) / 255.).astype(np.float32)

                # create meshgrid from the observed image
                coords = np.asarray(np.stack(np.meshgrid(np.linspace(0, W - 1, W), np.linspace(0, H - 1, H)), -1),
                                    dtype=int)

                # create sampling mask for interest region sampling strategy
                interest_regions = np.zeros((H, W, ), dtype=np.uint8)
                interest_regions[POI[:, 1], POI[:, 0]] = 1
                I = args.dil_iter
                interest_regions = cv2.dilate(interest_regions, np.ones(
                    (kernel_size, kernel_size), np.uint8), iterations=I)
                interest_regions = np.array(interest_regions, dtype=bool)
                interest_regions = coords[interest_regions]

                # not_POI -> contains all points except of POI
                coords = coords.reshape(H * W, 2)
                not_POI = set(tuple(point) for point in coords) - \
                    set(tuple(point) for point in POI)
                not_POI = np.array([list(point) for point in not_POI]).astype(int)



                # Create pose transformation model
                start_pose = torch.Tensor(start_pose).to(device)


                testsavedir = os.path.join(
                    output_dir, model_name, str(model_name)+"_"+str(index))
                os.makedirs(testsavedir, exist_ok=True)

                # imgs - array with images are used to create a video of optimization process


                if sampling_strategy == 'random':
                    rand_inds = np.random.choice(
                        coords.shape[0], size=batch_size, replace=False)
                    batch = coords[rand_inds]

                elif sampling_strategy == 'interest_points':
                    if POI.shape[0] >= batch_size:
                        rand_inds = np.random.choice(
                            POI.shape[0], size=batch_size, replace=False)
                        batch = POI[rand_inds]
                    else:
                        batch = np.zeros((batch_size, 2), dtype=np.int)
                        batch[:POI.shape[0]] = POI
                        rand_inds = np.random.choice(
                            not_POI.shape[0], size=batch_size-POI.shape[0], replace=False)
                        batch[POI.shape[0]:] = not_POI[rand_inds]

                elif sampling_strategy == 'interest_regions':
                    rand_inds = np.random.choice(
                        interest_regions.shape[0], size=batch_size, replace=False)
                    batch = interest_regions[rand_inds]

                else:
                    print('Unknown sampling strategy')
                    return

                target_s = obs_img[batch[:, 1], batch[:, 0]]
                target_s = torch.Tensor(target_s).to(device)
                # import pdb;pdb.set_trace()
                pose = start_pose.cpu()

                rays_o, rays_d = get_rays(
                    H, W, focal, pose)  # (H, W, 3), (H, W, 3)
                rays_o = rays_o[batch[:, 1], batch[:, 0]].to(device)  # (N_rand, 3)
                rays_d = rays_d[batch[:, 1], batch[:, 0]].to(device)  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)

                start_pose = torch.tensor(start_pose).to(device)
    
                raw_features, extras = generate_features(H, W, focal, chunk=args.chunk, rays=batch_rays,
                                                    verbose=False, retraw=True,
                                                    **render_kwargs)
                            
                points = torch.reshape(raw_features, (-1, raw_features.shape[-1]))
            
                # print(batch_rays[0,0,:])
                
                #remove last row of start pose for ease of use with nerfstudio
                input_image = torch.tensor(obs_img).unsqueeze(0).permute([0,3,1,2]).to(device)
                input_image = input_image.to(torch.float32)
                
                feature_image = dinov2_vits14_lc(input_image)
                
                #compute the loss and backpropagate
                loss = cost_based_matching(points, feature_image)
                
                
                loss.backward()
                
                optimizer.step()
            last_loss = loss.item()
            print(f'Epoch {epoch}, Loss: {last_loss}')

            
            

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    # train()
    train_feature_field()
