import torch
from torch import Tensor
from torchvision.transforms import ToPILImage
import nerfstudio
from nerfstudio.utils.colormaps import ColormapOptions
from nerfstudio.utils.eval_utils import eval_setup
import numpy as np
from nerfstudio.cameras.cameras import Cameras
from pathlib import PosixPath, Path
from typing import List
import json
from tqdm import tqdm
import os
import cv2
import json 
import imageio
from copy import deepcopy
from retrieval.loftr import LoFTR, default_cfg
import copy
import importlib




def show_mem_usage():
    usage = torch.cuda.mem_get_info()
    print("mem usage: ", 1 - usage[0]/usage[1])

def gen_camera(fov=np.pi/2.0, transform_matrix=None, im_size=(256,256)):
    """Generate a camera for generating rays
    fov is FOV in the x dimension.
    if no transform is supplied, it'll use a hardcoded one
    """
    
    image_width = im_size[0]
    image_height = im_size[1]
    pp_w = image_width / 2.0
    pp_h = image_height / 2.0
    focal_length = pp_h / np.tan(fov / 2.0) # since we are supplying the horizontal fov, this might need to change to pp_w
    intrinsics_matrix = torch.tensor([[focal_length, 0, pp_w], [0, focal_length, pp_h], [0, 0, 1]], dtype=torch.float32)
    fx = intrinsics_matrix[0, 0]
    fy = intrinsics_matrix[1, 1]

    if transform_matrix is None:
        transform_matrix = Tensor([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0.5]
        ])
    # else:
        # print(transform_matrix)

    return Cameras(transform_matrix, fx, fy, pp_w, pp_h)


def display_depth_image(o, filter=False): # input a depth tensor TODO: assert that its the correct input

    o = o.permute(2,0,1)

    # make the depth image look better (not sure how this affects other outputs)
    if filter:
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
    
    #note, comment if not using ssh
    image.save("depth_image.png")

    image.show()

def display_features_image(o):
    """Map a high dimension feature vector to an RGB color space and display it
    """

    a = nerfstudio.utils.colormaps.apply_colormap(o, colormap_options=ColormapOptions())
    a = a.permute(2,0,1)

    to_pil = ToPILImage()
    image = to_pil(a)
    
    #save image
    #note comment if not using  
    image.save("feature_image.png")

    image.show()


class CNNTrainingData():
    input_image_paths: List[PosixPath] = []
    feature_image_paths: List[PosixPath] = []
    transforms: List[Tensor] = []

    def __init__(self, model_path, file_path, force_overwrite=False):
        with open(file_path, 'r') as f:
            data = json.load(f)

            for frame in data['frames']:
                self.input_image_paths.append(PosixPath(frame['file_path']))
                self.transforms.append(Tensor(frame['transform_matrix']))

            # first check if the feature images already exist
            feature_im_dir = PosixPath('/'.join(PosixPath(file_path).parts[:-1])).joinpath('feature_images')
            if not feature_im_dir.exists():
                feature_im_dir.mkdir()
                print(f'Created {feature_im_dir}, proceeding to render feature images.')
            else:
                skip_render = True
                for filename in self.input_image_paths:
                    if not feature_im_dir.joinpath("features_" + filename.name).exists() and skip_render:
                        print("The directory exists, but is missing images. Proceeding to render feature images.")
                        skip_render = False

                    self.feature_image_paths.append(feature_im_dir.joinpath("features_" + filename.name))

                if force_overwrite:
                    print("Rendering feature images again")
                    skip_render = False

                if skip_render:
                    print(f'All images have already been generated in {feature_im_dir}. To generate again, use force_overwrite=True.')
                    return
                

            config, pipeline, _, _ = eval_setup(
                config_path=MODEL_PATH,
                test_mode='inference'
            )

            # tell the model's field to output the activations of a "intermediate"(hidden) layer
            pipeline.model.field.add_intermediate_outputs([1])

            # print(len(self.input_image_paths))
            # print(len(self.feature_image_paths))
            # print(len(self.transforms))

            print(f'Generating and saving feature images to {feature_im_dir}')

            # for in_path, out_path, tf in zip(self.input_image_paths, self.feature_image_paths, self.transforms):
            for i in tqdm(range(len(self.feature_image_paths))):
                out_path = self.feature_image_paths[i]
                tf = self.transforms[i]
                cam = gen_camera(fov=data['camera_angle_x'], transform_matrix=tf[0:3,:], im_size=(800,800))
                outputs = pipeline.model.get_outputs_for_camera(cam)

                a = nerfstudio.utils.colormaps.apply_colormap(outputs['layer1'], colormap_options=ColormapOptions())
                a = a.permute(2,0,1)

                to_pil = ToPILImage()
                image = to_pil(a)
                image.save(out_path)
                # print(f'save image to {out_path}')

                if i == 5:
                    print("cutting the loop short while testing (line ~145 in utils.py)")
                    import sys
                    sys.exit()
                    
            print("done")
            print(self.input_image_paths[0])
            # for path in self.input_image_paths:
                # ffn = path.name


if __name__ == "__main__":
    MODEL_PATH = Path('outputs/unnamed/nerfacto/2024-06-27_170932/config.yml')
    assert Path('outputs/unnamed/nerfacto/2024-06-27_170932/nerfstudio_models/step-000029999.ckpt').exists(), "The checkpoint file wasn't found."
    td = CNNTrainingData(MODEL_PATH, "./01Gorilla/transforms.json", force_overwrite=True)



'''
start by making the function save the images
but then make it save the serialized features
because thats what we actually want
'''
    

#Pad utilites after this line

def load_blender_ad(data_dir, model_name,  half_res, white_bkgd):
    # load train nerf images and poses
    meta = {}
    with open(os.path.join(data_dir, str(model_name), 'transforms.json'), 'r') as fp:
        meta["train"] = json.load(fp)
        imgs = []
        poses = []
        for frame in meta["train"]['frames']:
            fname = os.path.join(data_dir, str(model_name), frame['file_path'])
            img = imageio.imread(fname)
            img = (np.array(img) / 255.).astype(np.float32)
            if white_bkgd and img.shape[-1]==4:
                img = img[..., :3] * img[..., -1:] + (1. - img[..., -1:])
            else:
                img = img[..., :3]
            img = np.asarray(img*255, dtype=np.uint8)
            # img = tensorify(img)
            pose = (np.array(frame['transform_matrix']))
            pose = np.array(pose).astype(np.float32)
            imgs.append(img)
            poses.append(pose)
            
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['train']['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    # imgs=np.array(imgs)
    imgs = np.stack(imgs, 0)
    poses = np.array(poses)
    H, W = int(H), int(W)

    K = np.array([
        [focal, 0, 0.5*W],
        [0, focal, 0.5*H],
        [0, 0, 1]
    ])
    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.
        imgs_half_res = np.zeros((imgs.shape[0], H, W, 3))
        for i in range(len(imgs)):
            imgs_half_res[i] = cv2.resize(imgs[i], (W, H), interpolation=cv2.INTER_AREA)
        imgs=imgs_half_res.astype(np.uint8)
    return imgs,[H, W, focal],poses


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default='./data/nerf_synthetic/',
                        help='path to folder with synthetic or llff data')
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--model_name", type=str,
                        help='name of the nerf model')
    parser.add_argument("--output_dir", type=str, default='./output/',
                        help='where to store output images/videos')
    parser.add_argument("--ckpt_dir", type=str, default='./ckpts',
                        help='folder with saved checkpoints')
    parser.add_argument("--ckpt_name", type=str, 
                        help='name of ckpt')

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--chunk", type=int, default=1024*32,  # 1024*32
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64,  # 1024*64
                        help='number of pts sent through network in parallel, decrease if running out of memory')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=0.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels')

    # blender options
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    # llff options
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')

    # iNeRF options
    parser.add_argument("--obs_img_num", type=int, default=0,
                        help='Number of an observed image')
    parser.add_argument("--dil_iter", type=int, default=1,
                        help='Number of iterations of dilation process')
    parser.add_argument("--kernel_size", type=int, default=3,
                        help='Kernel size for dilation')
    parser.add_argument("--batch_size", type=int, default=2048,
                        help='Number of sampled rays per gradient step')
    parser.add_argument("--lrate", type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument("--sampling_strategy", type=str, default='random',
                        help='options: random / interest_point / interest_region')
    # parameters to define initial pose
    parser.add_argument("--delta_psi", type=float, default=0.0,
                        help='Rotate camera around x axis')
    parser.add_argument("--delta_phi", type=float, default=0.0,
                        help='Rotate camera around z axis')
    parser.add_argument("--delta_theta", type=float, default=0.0,
                        help='Rotate camera around y axis')
    parser.add_argument("--delta_t", type=float, default=0.0,
                        help='translation of camera (negative = zoom in)')
    # apply noise to observed image
    parser.add_argument("--noise", type=str, default='None',
                        help='options: gauss / salt / pepper / sp / poisson')
    parser.add_argument("--sigma", type=float, default=0.01,
                        help='var = sigma^2 of applied noise (variance = std)')
    parser.add_argument("--amount", type=float, default=0.05,
                        help='proportion of image pixels to replace with noise (used in ‘salt’, ‘pepper’, and ‘s&p)')
    parser.add_argument("--delta_brightness", type=float, default=0.0,
                        help='reduce/increase brightness of the observed image, value is in [-1...1]')
    
    parser.add_argument("--class_name", type=str, default='01Gorilla',
                        help='LEGO-3D anomaly class')
    
    parser.add_argument("--model_path", type=str, default='outputs/unnamed/nerfacto/2024-06-27_170932/config.yml',
                        help='path to the trained model')
    
    parser.add_argument("--epochs", type=int, default=1, help='number of epochs')
    
    

    return parser

def pose_retrieval_loftr(imgs,obs_img,poses):
    # The default config uses dual-softmax.
    # The outdoor and indoor models share the same config.
    # You can change the default values like thr and coarse_match_type.
    _default_cfg = deepcopy(default_cfg)
    _default_cfg['coarse']['temp_bug_fix'] = True  # set to False when using the old ckpt
    matcher = LoFTR(config=_default_cfg)
    matcher.load_state_dict(torch.load("./retrieval/model/indoor_ds_new.ckpt")['state_dict'])
    matcher = matcher.eval().cuda()
    if obs_img.shape[-1] == 3:
        query_img = cv2.cvtColor(obs_img, cv2.COLOR_RGB2GRAY)
    img0 = torch.from_numpy(query_img)[None][None].cuda() / 255.
    max_match=-1
    max_index=-1
    for i in range(len(imgs)):
        if imgs[i].shape[-1] == 3:
            gallery_img = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2GRAY)
        img1 = torch.from_numpy(gallery_img)[None][None].cuda() / 255.
        batch = {'image0': img0, 'image1': img1}

        # Inference with LoFTR and get prediction
        with torch.no_grad():
            matcher(batch)
            mkpts0 = batch['mkpts0_f'].cpu().numpy()
            mkpts1 = batch['mkpts1_f'].cpu().numpy()
            mconf = batch['mconf'].cpu().numpy()
        match_num=len(mconf)
        if match_num>max_match:
            max_match=match_num
            max_index=i
    return poses[max_index]


def rgb2bgr(img_rgb):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return img_bgr
def bgr2rgb(img_bgr):
    img_rgb=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
    return img_rgb

def show_img(title, img_rgb):  # img - rgb image
    img_bgr = rgb2bgr(img_rgb)
    cv2.imwrite(title, img_bgr)
    
def find_POI(img_rgb, DEBUG=False):  # img - RGB image in range 0...255
    img = np.copy(img_rgb)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints = sift.detect(img_gray, None)
    if DEBUG:
        img = cv2.drawKeypoints(img_gray, keypoints, img)
        show_img("Detected_points.png", img)
    xy = [keypoint.pt for keypoint in keypoints]
    xy = np.array(xy).astype(int)
    # Remove duplicate points
    xy_set = set(tuple(point) for point in xy)
    xy = np.array([list(point) for point in xy_set]).astype(int)
    return xy  # pixel coordinates
