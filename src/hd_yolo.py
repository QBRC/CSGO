import sys
sys.path.append('hd_wsi/')

from hd_wsi import run_patch_inference
from run_patch_inference import main
import argparse

import os

# find the path this file
SRC_DIR = os.path.realpath(os.path.dirname(__file__))

class yoloStandalone():
  def __init__(self, img_path, device, mpp, model_path=None):
    """A lightweight HD-Yolo used to extract patch-level nuclei information.

    Parameters
    ----------
    img_path : str
        The path of the image
    device : torch.device
        The PyTorch device destination
    mpp : int
        The MPP of the model. The resolution of the images used to train HD-Yolo
    model_path : str, optional
        Path of the HD-Yolo model pretrained weight, by default None
    """

    self.img_path = img_path
    self.device = device
    self.mpp = mpp
    self.model_path = model_path
  
  def args_init(self):
    """Initializes the HD-Yolo arguments 
    """
    args_yolo = argparse.Namespace()
    args_yolo.data_path = self.img_path
    args_yolo.output_dir = './'
    args_yolo.device = self.device 
    args_yolo.model = self.model_path
    args_yolo.meta_info = os.path.join(SRC_DIR, 'hd_wsi/meta_info.yaml')
    args_yolo.mpp = self.mpp
    args_yolo.box_only = False
    args_yolo.export_text = False

    self.args_yolo = args_yolo

    return args_yolo

  def run_inference(self):
    """Wrapper for the original `run_patch_inference`
    Notes
    -----
    For further reference, please consult the original HD-Yolo page https://github.com/impromptuRong/hd_wsi
    """
    # 0 is the background (non-nuclei) pixel value 
    nuclei_pred, patch = run_patch_inference.main(self.args_yolo, self.device)
    return nuclei_pred, patch