import pytest
from models import CSGO
# export PATH="/project/DPDS/Xiao_lab/shared/deep_learning_SW_RR/cell_segmentation/CSGO/src:$PATH"

# export PYTHONPATH=/project/DPDS/Xiao_lab/shared/deep_learning_SW_RR/cell_segmentation/CSGO/src:$PYTHONPATH
def test_csgo_init_no_gpu():
  cell_seg_go = CSGO()
  assert cell_seg_go.device == None

def test_csgo_init_with_gpu():
  cell_seg_go = CSGO(gpu=True)
  assert cell_seg_go.device == 0