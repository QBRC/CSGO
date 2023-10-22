# add the following to command line before running pytest
# export PYTHONPATH=/project/DPDS/Xiao_lab/shared/deep_learning_SW_RR/cell_segmentation/CSGO/src:$PYTHONPATH
# see pytest.ini for CI configs
import pytest
from models import CSGO

def test_csgo_init_no_gpu():
  cell_seg_go = CSGO()
  assert cell_seg_go.device == None

def test_csgo_init_with_gpu():
  cell_seg_go = CSGO(gpu=True)
  assert cell_seg_go.device == 0

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


  