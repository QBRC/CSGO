# add the following to command line before running pytest
# export PYTHONPATH=/project/DPDS/Xiao_lab/shared/deep_learning_SW_RR/cell_segmentation/CSGO/src:$PYTHONPATH
# see pytest.ini for CI configs

from models import CSGO

def test_csgo_init_no_gpu():
  cell_seg_go = CSGO()
  assert cell_seg_go.device == None

def test_csgo_init_with_gpu():
  cell_seg_go = CSGO(gpu=True)
  assert cell_seg_go.device == 0