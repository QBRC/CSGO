# add the following to command line before running pytest
# export PYTHONPATH=/project/DPDS/Xiao_lab/shared/deep_learning_SW_RR/cell_segmentation/CSGO/src:$PYTHONPATH
# see pytest.ini for CI configs
import pytest
from models import CSGO
from torch_models import UNet, SoftDiceLoss
import numpy as np
import os
import skimage

YOLO_PATH = './src/pretrained_weights/lung_best.float16.torchscript.pt'
UNET_PATH = './src/pretrained_weights/epoch_190.pt'
IMG_PATH = './example_patches/TCGA-UB-AA0V-01Z-00-DX1.FB59AF14-B425-488D-94FD-E999D4057468.png'


def test_csgo_init_no_gpu():
  cell_seg_go = CSGO() # GPU false by default
  assert cell_seg_go.device.type == 'cpu'

def test_csgo_init_with_gpu():
  cell_seg_go = CSGO(gpu=True)
  if cell_seg_go.device.type != 'cuda':
    pytest.skip("Indicated GPU use, but no GPU found")

  assert cell_seg_go.device.type == 'cuda'

@pytest.fixture
def csgo_for_tests_shared():
  'create fixture to test multiple mpp conversion scenarios'
  cell_seg_go = CSGO(yolo_path=YOLO_PATH, unet_path=UNET_PATH, zoom=40, mpp=0.25)
  return cell_seg_go

def test_resolution_mpp_convertion(csgo_for_tests_shared):
  # integer conversions
  new_mpp = csgo_for_tests_shared.convert_resolution_to_mpp(80)
  assert new_mpp == 0.125

  # float mpp check
  new_mpp = csgo_for_tests_shared.convert_resolution_to_mpp(60)
  expected_mpp = 0.25 / (60/40)
  assert new_mpp == pytest.approx(expected_mpp)

def test_unet_init(csgo_for_tests_shared):
  csgo_for_tests_shared.unet_init()
  
  model = csgo_for_tests_shared.model
  assert isinstance(model, UNet)

def test_unet_seg(csgo_for_tests_shared):
  '''
  Test if UNet is producing the expected outputs.
  This only tests the software portion (i.e. output sizes, values), visual inspection needs to be conducted seprately.
  '''
  csgo_for_tests_shared.unet_init()

  rand_img = np.random.uniform(low=0, high=255, size=(512, 512, 3))
  
  membrane_pred = csgo_for_tests_shared.membrane_detection(rand_img, patch_mpp=0.25)
  min_check = membrane_pred >= 0
  max_check = membrane_pred <= 255
  assert np.all(min_check) and np.all(max_check)


def test_watershed(csgo_for_tests_shared):
  TEST_DIR = os.path.realpath(os.path.dirname(__file__))
  nuclei_masks = skimage.io.imread(os.path.join(TEST_DIR, 'test_nuclei_masks.tiff'))
  membrane_masks = skimage.io.imread(os.path.join(TEST_DIR, 'test_membrane_masks.tiff'))

  cell_seg = csgo_for_tests_shared.watershed(nuclei_masks, membrane_masks, cell_size = 40)
  
  assert isinstance(cell_seg, np.ndarray)
  assert cell_seg.shape == nuclei_masks.shape

def test_segmentation(csgo_for_tests_shared):
  '''
  light weight segmentation test. Visual inspection is needed.
  '''
  try:
    res = csgo_for_tests_shared.segment(IMG_PATH, cell_size=40)
    assert isinstance(res, np.ndarray)

  # generic errors. Recommend code review if skipped
  except (FileNotFoundError, ValueError): # specific to GitHub CI
    pytest.skip('No Yolo weights uploaded to GitHub')


  