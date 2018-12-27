import os
import sys
import json
import numpy as np
from src.base_models.base_model import BaseModel
import torch
nn = torch.nn

WORK_DIR = os.environ['ROOT_DIR']
sys.path.append(WORK_DIR)


class Model(BaseModel):
    def __init__(self, model_config, hyperparam_config, env_config):
        super(Model, self).__init__(model_config=model_config,
                                    hyperparam_config=hyperparam_config,
                                    env_config=env_config)
