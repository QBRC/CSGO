from hd_yolo import yoloStandalone
import numpy as np
import torch
import os
import pytest

import logging
log = logging.getLogger(__name__)

IMG_PATH = './example_patches/TCGA-UB-AA0V-01Z-00-DX1.FB59AF14-B425-488D-94FD-E999D4057468.png'
YOLO_PATH = './src/pretrained_weights/lung_best.float16.torchscript.pt'
DEVICE = torch.device('cpu')


def test_yolo_init():
  '''
  Test if HD-Yolo can be intialized
  '''
  yolo = yoloStandalone(IMG_PATH, device=DEVICE, mpp=0.25, model_path=YOLO_PATH)
  args = yolo.args_init()
  assert args.box_only == False
  assert args.data_path == './example_patches/TCGA-UB-AA0V-01Z-00-DX1.FB59AF14-B425-488D-94FD-E999D4057468.png'

  # check some class attributes
  class_args = yolo.args_yolo
  assert class_args.mpp == 0.25
  assert class_args.device.type == 'cpu' 
  

def test_yolo_inference():
  '''
  Test if HD-Yolo can produce the output as intended.
  '''

  yolo = yoloStandalone(IMG_PATH, device=DEVICE, mpp=0.25, model_path=YOLO_PATH)
  args = yolo.args_init()

  # on GitHub CI, skip since no model weights available
  if not os.path.exists(args.model):
    log.warning("No model weights found. Testing skipped but separate test should be conducted.")
    pytest.skip('No model weights found.')
    
  nuclei_pred, patch = yolo.run_inference()
  
  # check output types
  assert isinstance(nuclei_pred, np.ndarray)
  assert isinstance(patch, np.ndarray)

  # nuclei_pred outputs with shape (W x H x Channel). Only care about the WxH
  assert nuclei_pred.shape[:2] == patch.shape[:2]
