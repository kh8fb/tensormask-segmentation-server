"""
Interface for loading the supported image segmentation models.
"""

import cv2
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model
import numpy as np
from PIL import Image
import tensormask
from tensormask import add_tensormask_config
import torch
import torchvision.transforms as transforms


def load_tensormask_model(model_path, cfg_path, device):
    """
    Load the pretrained TensorMask model states and prepare the model for image segmentation.

    Paramters
    ---------
    model_path: str
        Path to the pretrained model states binary file.
    cfg_path: str
        Path to the model's Configuration file.  Located in the .configs folder.
    device: torch.device
        Device to load the model on.

    Returns
    -------
    model: TensorMask
        Model with the loaded pretrained states.
    """
    # set up model config
    cfg = get_cfg()
    tensormask.add_tensormask_config(cfg)
    cfg.merge_from_file(cfg_path)
    model = build_model(cfg)

    # load the model weights
    DetectionCheckpointer(model).load(model_path)
    model.eval()

    return model


def load_models(device, tensormask_path, cfg_path):
    """
    Load the model return them in a dictionary.

    Parameters
    ----------
    device: torch.device
        Device to load the model on.
    tensormask_path: str or None
        Path to the pretrained TensorMask model states binary file.
    cfg_path: str
        Path to the model's configuration file.  Located in the .configs folder.

    Returns
    -------
    model_dict: dict
        Dictionary storing the model and model name.
        Current keys are 'model_name', 'model'.
    """
    if tensormask_path is not None:
        tensormask_model = load_bit_model(str(tensormask_path), str(cfg_path), device)
        return {"model_name": "tensormask", "model": tensormask_model}
    # add additional models here
    else:
        return None
