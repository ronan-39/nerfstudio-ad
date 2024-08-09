import nerfstudio
import torch
import numpy as np
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

import utils

def main():
    # MODEL_PATH = Path('outputs/unnamed/nerfacto/2024-06-27_170932/config.yml')
    # assert Path('outputs/unnamed/nerfacto/2024-06-27_170932/nerfstudio_models/step-000029999.ckpt').exists(), "The checkpoint file wasn't found."
    
    # config, pipeline, _, _ = eval_setup(
    #     config_path=MODEL_PATH,
    #     test_mode='inference'
    # )

    # # generate a camera with simple params pointed at the center of the scene
    # cam = utils.gen_camera()

    # # tell the model's field to output the activations of a "intermediate"(hidden) layer
    # pipeline.model.field.add_intermediate_outputs([0,1,2]) # TODO some bug where if i dont add all three of these, the output of layer1 is wrong

    # # get outputs of the NeRF from the camera's POV
    # outputs = pipeline.model.get_outputs_for_camera(cam)

    # print(outputs.keys())

    # print(outputs['layer0'].shape)
    # print(outputs['layer1'].shape)
    # print(outputs['layer2'].shape)
    # import time

    # # torch.summary()
    # # print(pipeline.model.field.mlp_head)

    # utils.display_features_image(outputs['layer0'])
    # time.sleep(1)
    # utils.display_features_image(outputs['layer1'])
    # time.sleep(1)
    # utils.display_features_image(outputs['layer2'])
    # time.sleep(1)
    # utils.display_features_image(outputs['rgb'])
    # # utils.display_depth_image(outputs['depth'], filter=False)
    # # utils.display_depth_image(outputs['depth'], filter=True)

    # # print(type(MODEL_PATH))

    # # training_data = utils.CNNTrainingData("./01Gorilla/transforms.json")
    # # print(training_data)

    # a list of paths to the config.yml files generated by ns-train, and the file prefix for the generated training data
    nerfs = [
        ['gorilla', '~/Documents/datasets/MAD-Sim/01Gorilla/outputs/gorilla_torch/nerfacto/2024-08-09_123452/config.yml'],
        # ['unicorn', ''],
        ['mallard', '~/Documents/datasets/MAD-Sim/03Mallard/outputs/mallard_torch/nerfacto/2024-08-08_141411/config.yml'],
        ['turtle',  '~/Documents/datasets/MAD-Sim/04Turtle/outputs/turtle_torch/nerfacto/2024-08-08_154430/config.yml'],
        ['whale',   '~/Documents/datasets/MAD-Sim/05Whale/outputs/whale_torch/nerfacto/2024-08-08_170849/config.yml'],
        ['bird',    '~/Documents/datasets/MAD-Sim/06Bird/outputs/bird_torch/nerfacto/2024-08-08_174210/config.yml'],
        ['owl',     '~/Documents/datasets/MAD-Sim/07Owl/outputs/owl_torch/nerfacto/2024-08-08_203543/config.yml'],
        ['swan',    '~']
    ]

    for nerf in nerfs:
        utils.create_training_data_from_nerf(nerf[1], './im_encoder', prefix=nerf[0], num_images=16)


main()
