# add the following to command line before running pytest
# export PYTHONPATH=/project/DPDS/Xiao_lab/shared/deep_learning_SW_RR/cell_segmentation/CSGO/src:$PYTHONPATH
# see pytest.ini for CI configs
import pytest
from models import CSGO

def test_csgo_init_no_gpu():
  cell_seg_go = CSGO() # GPU false by default
  assert cell_seg_go.device.type == 'cpu'

def test_csgo_init_with_gpu():
  cell_seg_go = CSGO(gpu=True)
  if cell_seg_go.device.type != 'cuda':
    pytest.skip("Indicated GPU use, but no GPU found")

  assert cell_seg_go.device.type == 'cuda'

@pytest.fixture
def csgo_no_gpu():
  'create fixture to test multiple mpp conversion scenarios'
  cell_seg_go = CSGO(gpu=False, zoom=40, mpp=0.25)
  return cell_seg_go


@pytest.fixture
def csgo_for_mpp_check():
  'create fixture to test multiple mpp conversion scenarios'
  cell_seg_go = CSGO(zoom=40, mpp=0.25)
  return cell_seg_go

def test_resolution_mpp_convertion(csgo_for_mpp_check):
  # integer conversions
  new_mpp = csgo_for_mpp_check.convert_resolution_to_mpp(80)
  assert new_mpp == 0.125

  # float mpp check
  new_mpp = csgo_for_mpp_check.convert_resolution_to_mpp(60)
  expected_mpp = 0.25 / (60/40)
  assert new_mpp == pytest.approx(expected_mpp)


# def test_yolo_init(csgo_no_gpu):
  # 'test if yolo can be run'
  # img_path = '../for_dev_only/TCGA-UB-AA0V-01Z-00-DX1.FB59AF14-B425-488D-94FD-E999D4057468.png'
  # args = csgo_no_gpu.run_yolo(img_path, mpp=0.25)




  