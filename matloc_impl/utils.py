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

    if transform_matrix == None:
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

    image.show()

def display_features_image(o):
    """Map a high dimension feature vector to an RGB color space and display it
    """

    a = nerfstudio.utils.colormaps.apply_colormap(o, colormap_options=ColormapOptions())
    a = a.permute(2,0,1)

    to_pil = ToPILImage()
    image = to_pil(a)

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