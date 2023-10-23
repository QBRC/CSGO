import sys
sys.path.append('hd_wsi/')

from hd_wsi import run_patch_inference
from run_patch_inference import main
import argparse

import os

# find the path this file
SRC_DIR = os.path.realpath(os.path.dirname(__file__))

class yolo_standalone():
  def __init__(self, img_path, device, mpp):
    self.img_path = img_path
    self.device = device
    self.mpp = mpp
  
  def args_init(self):
    args_yolo = argparse.Namespace()
    args_yolo.data_path = self.img_path
    args_yolo.output_dir = './'
    args_yolo.device = self.device 
    args_yolo.model = os.path.join(SRC_DIR, 'for_dev_only/pretrained_weights/lung_best.float16.torchscript.pt')
    args_yolo.meta_info = os.path.join(SRC_DIR, 'hd_wsi/meta_info.yaml')
    args_yolo.mpp = self.mpp
    args_yolo.box_only = False
    args_yolo.export_text = False

    self.args_yolo = args_yolo

    return args_yolo

  def run_inference(self):
    # 0 is the background (non-nuclei) pixel value 
    nuclei_pred, patch = run_patch_inference.main(self.args_yolo, self.device)
    return nuclei_pred, patch