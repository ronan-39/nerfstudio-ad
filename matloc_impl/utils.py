import torch
from torch import Tensor
from torchvision.transforms import ToPILImage
import torchvision
import nerfstudio
from nerfstudio.utils.colormaps import ColormapOptions
from nerfstudio.utils.eval_utils import eval_setup
import numpy as np
from nerfstudio.cameras.cameras import Cameras
from pathlib import PosixPath, Path
from typing import List
import json
from tqdm import tqdm
import math
import pickle
import yaml
import glob

def show_mem_usage():
    usage = torch.cuda.mem_get_info()
    print("mem usage: ", 1 - usage[0]/usage[1])

def gen_camera(fov=np.pi/2.0, transform_matrix=None, im_size=(512,307)):
    """Generate a camera for generating rays
    fov is FOV in the x dimension.
    if no transform is supplied, it'll use a hardcoded one
    """

    # fov = 1.3089969389957472 # default viewer fov
    # fov = np.pi/2.0
    # print("fov", fov)
    # fov = 0.69111 by default
    # fov = 0.69111
    # print("fov:", fov)

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

    # transform_matrix = Tensor([[[-7.0711e-01, -4.0825e-01,  5.7735e-01,  3.0000e-01],
    #      [ 7.0711e-01, -4.0825e-01,  5.7735e-01,  3.0000e-01],
    #      [-1.6653e-16,  8.1650e-01,  5.7735e-01,  3.0000e-01]]])

    # return Cameras(transform_matrix, fx, fy, pp_w, pp_h)
    return Cameras(
        fx=fx,
        fy=fy,
        cx=pp_w,
        cy=pp_h,
        camera_to_worlds=transform_matrix.float()
    )

    # return Cameras(camera_to_worlds=Tensor([[[-7.0711e-01, -4.0825e-01,  5.7735e-01,  3.0000e-01],
        #  [ 7.0711e-01, -4.0825e-01,  5.7735e-01,  3.0000e-01],
        #  [-1.6653e-16,  8.1650e-01,  5.7735e-01,  3.0000e-01]]]), fx = Tensor([[200.0415]]), fy=Tensor([[200.0451]]), cx=Tensor([[256.]]), cy=Tensor([[153.500]]), width=512, height=307)

"""
Cameras(camera_to_worlds=tensor([[[-7.0711e-01, -4.0825e-01,  5.7735e-01,  3.0000e-01],
         [ 7.0711e-01, -4.0825e-01,  5.7735e-01,  3.0000e-01],
         [-1.6653e-16,  8.1650e-01,  5.7735e-01,  3.0000e-01]]],
       device='cuda:0'), fx=tensor([[200.0451]], device='cuda:0'), fy=tensor([[200.0451]], device='cuda:0'), cx=tensor([[256.]], device='cuda:0'), cy=tensor([[153.5000]], device='cuda:0'), width=tensor([[512]], device='cuda:0'), height=tensor([[307]], device='cuda:0'), distortion_params=None, camera_type=tensor([[1]], device='cuda:0'), times=tensor([[0.]], device='cuda:0'), metadata=None)
"""


def display_depth_image(o, filter=False): # input a depth tensor TODO: assert that its the correct input

    if filter == False:
        o = o.permute(2,0,1)

    # make the depth image look better (not sure how this affects other outputs)
    if filter:
        _max = o.max()
        _min = o.min()

        o -= _min.item()
        o /= _max.item()
        o = np.power(o, 1/4) # tune the exponent per NeRF, 1/4 is good for the gorilla
        o *= 255.0
        o = np.where( o > 39.0, 255.0, o)

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


def points_on_sphere(radius, phi_divs=14, theta_divs=14):
    """Generate a list of transforms that place a camera on the surface of the sphere pointing toward the origin
    """

    tfs = []
    
    for t in range(theta_divs):
        theta = t/theta_divs * 1 * np.pi + (1/theta_divs * np.pi)
        for p in range(phi_divs):
            phi = p/phi_divs * 2 * np.pi + (1/phi_divs * 2 * np.pi)

            x = radius * np.sin(theta) * np.cos(phi)
            y = radius * np.sin(theta) * np.sin(phi)
            z = radius * np.cos(theta)

            t = torch.tensor([x, y, z])

            z_basis = t/ torch.norm(t)
            second_basis = -np.cross(t, torch.tensor([0,0,-1]))
            second_basis /= np.linalg.norm(second_basis)
            second_basis = torch.from_numpy(second_basis)
            if torch.allclose(z_basis.float(), torch.tensor([0,0,1]).float()):
                second_basis = torch.tensor([0,1,0])
            third_basis = torch.from_numpy(np.cross(z_basis, second_basis))

            r = torch.from_numpy(np.identity(3))
            r[0:3, 2] = z_basis
            r[0:3, 1] = -third_basis
            r[0:3, 0] = second_basis

            tf = torch.cat([r, t.view(-1,1)], dim=1)
            tf = torch.cat([tf, torch.tensor([[0,0,0,1]])])

            tfs.append(tf) # todo: convert to torch tensor here, use np arrays otherwise

    return tfs


class CNNTrainingData():
    rgb_image_paths: List[Path] = []
    feature_paths: List[Path] = []
    feature_image_paths: List[Path] = []
    transforms: List[Tensor] = []

    def __init__(self, model_path, file_path, force_overwrite=False):
        print("creating CNN training data")
        self.init_new(model_path, file_path, force_overwrite=force_overwrite, render_feature_images=True)

    def init_new(self, model_path, _file_path, num_images=14*14, force_overwrite=False, render_feature_images=False):
        if type(num_images) != int or num_images != math.isqrt(num_images) ** 2:
            raise Exception("num_images must be square")
        
        output_dir = Path(('/').join(model_path.parts[0:2])).joinpath("feature_image_pairs")

        should_render = False

        if not output_dir.exists():
            output_dir.mkdir()
            output_dir.joinpath("rgb").mkdir()
            output_dir.joinpath("features").mkdir()
            should_render = True
            print(f'Created {output_dir}, proceeding to render image pairs')
        
        if not should_render and force_overwrite:
            print("Force overwrite is true, so rendering image pairs")
            should_render = True

        for i in range(num_images):
            rgb_filepath = output_dir.joinpath('rgb').joinpath(f'rgb_{i}.png')
            feat_filepath = output_dir.joinpath('features').joinpath(f'feat_{i}.feature')
            feat_image_filepath = output_dir.joinpath('features').joinpath(f'feat_{i}.png')

            filepaths = [rgb_filepath, feat_filepath]

            if render_feature_images:
                filepaths.append(feat_image_filepath)

            if not should_render:
                if not all([p.exists() for p in filepaths]):
                    print('Output directories exists, but files are missing. Rendering image pairs')
                    should_render=True

            self.rgb_image_paths.append(rgb_filepath)
            self.feature_paths.append(feat_filepath)

            if render_feature_images:
                self.feature_image_paths.append(feat_image_filepath)


        if not should_render:
            print("shouldnt render. returning")
            return
        
        self.transforms = points_on_sphere(0.6, phi_divs=math.isqrt(num_images), theta_divs=math.isqrt(num_images))

        config, pipeline, _, _ = eval_setup(
            config_path=MODEL_PATH,
            test_mode='inference'
        )

        pipeline.model.field.add_intermediate_outputs([0,1,2])

        for i in tqdm(range(len(self.transforms))):
            tf = self.transforms[i]
            cam = gen_camera(transform_matrix=tf[0:3,:], im_size=(224,224))
            cam = cam.to(pipeline.model.device)
            assert isinstance(cam, Cameras)
            outputs = pipeline.model.get_outputs_for_camera(cam)

            a = nerfstudio.utils.colormaps.apply_colormap(outputs['rgb'], colormap_options=ColormapOptions())
            rgb_image = a.permute(2,0,1)

            to_pil = ToPILImage()
            image = to_pil(rgb_image)
            image.save(self.rgb_image_paths[i])

            a = nerfstudio.utils.colormaps.apply_colormap(outputs['layer1'], colormap_options=ColormapOptions())
            a = a.permute(2,0,1)

            to_pil = ToPILImage()
            image = to_pil(a)
            image.save(self.feature_image_paths[i])

            with open(self.feature_paths[i], "wb") as outfile:
                pickle.dump(outputs['layer1'], outfile)

            # if i == 5:
            #     print(self.feature_paths[i])
            #     print(outputs['layer1'].shape)
            #     print("cutting short")
            #     import sys
            #     sys.exit()


    def init_old(self, model_path, file_path, force_overwrite=False):
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
                cam = gen_camera(fov=data['camera_angle_x']/(np.pi), transform_matrix=tf[0:3,:], im_size=(800,800))
                cam = cam.to(pipeline.model.device)
                assert isinstance(cam, Cameras)
                outputs = pipeline.model.get_outputs_for_camera(cam)

                a = nerfstudio.utils.colormaps.apply_colormap(outputs['rgb'], colormap_options=ColormapOptions())
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


def create_training_data_from_nerf(nerf_dir, output_dir, num_images=14*14, force_overwrite=False, prefix=None, debug_labels=False):
    '''
    nerf: a directory(represented by a string) that points to the .ckpt file generated by a trained nerf
    output_dir: the directory to place the training data pairs
    '''
    if prefix is None:
        print("Right now, you need to include a prefix. This prefix will be in the file name of all the training data")
        return

    if '~/' in nerf_dir:
        nerf_dir = nerf_dir.replace('~', str(Path.home()))

    # first, check if the nerf exists
    nerf_p = Path(nerf_dir)
    if not nerf_p.exists():
        print('create_training_data_from_nerf() has been passed an invalid path to a nerf\'s config.yml')
        print(nerf_p)
        return
    
    # make sure its using torch
    config = yaml.load(nerf_p.read_text(), Loader=yaml.Loader)
    if config.pipeline.model.implementation != 'torch':
        print("Cannot create training data from a NeRF trained with TCNN")
        print(nerf_p)
        return

    # next, check if training data has already been created
    out_p = Path(output_dir)
    if not out_p.exists():
        print("The specified output directory does not exist. to prevent creating training data in unintended locations, this function won't automatically create directories.")
    
    image_names = [Path(prefix + "_" + str(i) + ".png") for i in range(num_images)]
    label_img_names = [Path(prefix + "_" + str(i) + "_label.png") for i in range(num_images)]
    label_names = [Path(prefix + "_" + str(i) + "_label.feature") for i in range(num_images)]

    if all([name.exists() for name in image_names]) and all([name.exists() for name in label_names]):
        print("It seems all the training data already exist. Won't overwrite unless 'force_overwrite' is true")
        if not force_overwrite:
            return
    else:
        print("Not all the training data exists")

    if all([name.exists() for name in label_img_names]) and debug_labels:
        print("It seems all the debug label images already exist. Won't overwrite unless 'force_overwrite' is true")

    # create the training data
    transforms = points_on_sphere(0.6, phi_divs=math.isqrt(num_images), theta_divs=math.isqrt(num_images))

    print("nerf_p:", nerf_p)
    config, pipeline, _, _ = eval_setup(
        config_path=nerf_p,
        test_mode='inference'
    )

    pipeline.model.field.add_intermediate_outputs([0,1,2])

    print("Creating training data for", prefix)
    for i in tqdm(range(len(transforms))):
        image_path = out_p.joinpath('train', 'images', image_names[i])
        label_image_path = out_p.joinpath('train', 'labels', label_img_names[i])
        label_path = out_p.joinpath('train', 'labels', label_names[i])

        tf = transforms[i]
        cam = gen_camera(transform_matrix=tf[0:3,:], im_size=(224,224))
        cam = cam.to(pipeline.model.device)
        assert isinstance(cam, Cameras)
        outputs = pipeline.model.get_outputs_for_camera(cam)

        a = nerfstudio.utils.colormaps.apply_colormap(outputs['rgb'], colormap_options=ColormapOptions())
        rgb_image = a.permute(2,0,1)

        to_pil = ToPILImage()
        image = to_pil(rgb_image)
        image.save(image_path)

        a = nerfstudio.utils.colormaps.apply_colormap(outputs['layer1'], colormap_options=ColormapOptions())
        a = a.permute(2,0,1)

        to_pil = ToPILImage()
        image = to_pil(a)
        image.save(label_image_path)

        with open(label_path, "wb") as outfile:
            pickle.dump(outputs['layer1'], outfile)

        # print("save", image_path)
        # print("save", label_path)
        # print("save", label_image_path)

def generate_masks(dir, threshold):
    """
    generate a mask for the white parts of an image. this is specifically for use with MAD NeRFs, since the
    RGB background is a mostly clean white, while the feature images have a lot of noise in the background.
    
    path: directory where all the rgb images are stored
    tolerance: maximum intensity of masked area
    """
    if '~/' in dir:
        dir = dir.replace('~', str(Path.home()))

    path = Path(dir)
    output_path = path.joinpath("masks")

    if not path.exists():
        raise Exception("directory does not exist")
    
    if not output_path.exists():
        output_path.mkdir()
    
    to_pil = ToPILImage()
    for filepath in glob.iglob(str(path) + "/*.png"):
        im = torchvision.io.read_image(filepath, mode=torchvision.io.ImageReadMode.GRAY)
        mask = (im <= threshold).float()

        mask_name = filepath.split('/')[-1].split('.')[0] + "_mask.png"
        
        image = to_pil(mask)
        image.save(output_path.joinpath(mask_name))


if __name__ == "__main__":
    MODEL_PATH = Path('outputs/unnamed/nerfacto/2024-06-27_170932/config.yml')
    assert Path('outputs/unnamed/nerfacto/2024-06-27_170932/nerfstudio_models/step-000029999.ckpt').exists(), "The checkpoint file wasn't found."

    # MODEL_PATH = Path('outputs/01Gorilla/nerfacto/2024-06-21_160959/config.yml') # tcnn model
    # assert Path('outputs/unnamed/nerfacto/2024-06-27_170932/nerfstudio_models/step-000029999.ckpt').exists(), "The checkpoint file wasn't found."
    # td = CNNTrainingData(MODEL_PATH, "./01Gorilla/transforms.json", force_overwrite=False)

    # points_on_sphere(1.0)

    # i have to be in ./02Unicorn to make this bit work... this is so messy but i swear ill fix it later
    # MODEL_PATH = Path('outputs/unnamed/nerfacto/2024-07-25_235627/config.yml')
    # MODEL_PATH = Path('outputs/unicorn_torch/nerfacto/2024-07-26_003033/config.yml')
    # assert MODEL_PATH.exists(), "The checkpoint file wasn't found"
    # assert Path('02Unicorn/outputs/unnamed/nerfacto/2024-07-25_235627/nerfstudio_models/step-000029999.ckpt').exists(), "The checkpoint file wasn't found."
    CNNTrainingData(MODEL_PATH, "./this_shouldnt_exist", force_overwrite=True)

'''
start by making the function save the images
but then make it save the serialized features
because thats what we actually want
'''