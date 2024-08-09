"""
checking out mipnerf, interested to see the network architecture.
"""

from pathlib import Path
import nerfstudio
from nerfstudio.utils.eval_utils import eval_setup
import torch

def main():
    model_path = Path.home().joinpath("Documents/datasets/MAD-Sim/01Gorilla/outputs/gorilla_mip/mipnerf/2024-08-09_160505/config.yml")

    config, pipeline, _, _ = eval_setup(
        config_path=model_path,
        test_mode='inference'
    )

    print(pipeline.model)

if __name__ == "__main__":
    main()