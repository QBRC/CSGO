from hd_yolo import yolo_standalone
import numpy as np


IMG_PATH = './src/for_dev_only/TCGA-UB-AA0V-01Z-00-DX1.FB59AF14-B425-488D-94FD-E999D4057468.png'
DEVICE = 0


def test_yolo_init():
  '''
  Test if HD-Yolo can be intialized
  '''
  yolo = yolo_standalone(IMG_PATH, device=DEVICE, mpp=0.25)
  args = yolo.args_init()
  assert args.box_only == False
  assert args.data_path == './src/for_dev_only/TCGA-UB-AA0V-01Z-00-DX1.FB59AF14-B425-488D-94FD-E999D4057468.png'

  # check some class attributes
  class_args = yolo.args_yolo
  assert class_args.mpp == 0.25
  assert class_args.device == DEVICE 
  

def test_yolo_inference():
  '''
  Test if HD-Yolo can produce the output as intended.
  '''
  yolo = yolo_standalone(IMG_PATH, device=DEVICE, mpp=0.25)
  args = yolo.args_init()
  nuclei_pred, patch = yolo.run_inference()
  
  # check output types
  assert isinstance(nuclei_pred, np.ndarray)
  assert isinstance(patch, np.ndarray)

  # nuclei_pred outputs with shape (W x H x Channel). Only care about the WxH
  assert nuclei_pred.shape[:2] == patch.shape[:2]
