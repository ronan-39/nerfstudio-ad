import os
import torch
from tqdm import tqdm
import time
import imageio
import numpy as np
import torch.nn.functional as F
#from torchsearchsorted import searchsorted
DEBUG = False

def cost_based_matching(NFF_Features, Image_Feature):
    
    
    #size of NFF_Features is [N_points, F]
    #size of Image_Feature is [1, F]
    
    #First distance, distance between each point and the image feature
    #size of distance is [N_points, 1]
    feature_distance = torch.sum(torch.norm(NFF_Features - Image_Feature, dim=1))
    
    
    #second distance, distance between the 3d point and the feature in 3d
    #size of distance is [N_points, 1]
    #3d point is the first 3 elements of NFF_Features
    #3d feature is the first 3 elements of Image_Feature
    
    
    #last metric, how common is the image feature in the point cloud
    #size of distance is [1,1]
    #number of points that have the same feature as the image feature
    
    
    
    
    total_loss = feature_distance
    
    
    
    return total_loss



# Ray helpers
def get_rays(H, W, focal, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d
