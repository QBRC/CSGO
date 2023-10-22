import sys
sys.path.append('hd_wsi/')

from hd_wsi import run_patch_inference
from run_patch_inference import main
import argparse

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
    args_yolo.model = '/project/DPDS/Xiao_lab/shared/deep_learning_SW_RR/cell_segmentation/watershed/hd_wsi_models/lung_best.float16.torchscript.pt'
    args_yolo.meta_info = './hd_wsi/meta_info.yaml'
    args_yolo.mpp = self.mpp
    args_yolo.box_only = False
    args_yolo.export_text = False

    return args_yolo

