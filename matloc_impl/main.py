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

    print("depth shape:", outputs['rgb'].shape)
    utils.display_features_image(outputs['layer1'])
    # utils.display_depth_image(outputs['rgb'][:,:,1])


main()

# utils.display_features_image(1)