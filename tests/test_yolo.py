from hd_yolo import yolo_standalone

IMG_PATH = '../for_dev_only/TCGA-UB-AA0V-01Z-00-DX1.FB59AF14-B425-488D-94FD-E999D4057468.png'
DEVICE = 0


def test_yolo_init():
  '''
  Test if HD-Yolo can be intialized
  '''
  yolo = yolo_standalone(IMG_PATH, device=DEVICE, mpp=0.25)
  args = yolo.args_init()
  assert args.box_only == False
  assert args.data_path == '../for_dev_only/TCGA-UB-AA0V-01Z-00-DX1.FB59AF14-B425-488D-94FD-E999D4057468.png'