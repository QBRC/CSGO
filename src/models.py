from hd_yolo import yolo_standalone
import argparse
import torch

class CSGO():
  def __init__(self, gpu=False, save=False, zoom=40, mpp=0.25):
    """
    Define model level info.
    zoom: 1st data point to convert resolution and mpp
    mpp: 2nd data point to convert resolution and mpp
    standard equipment places 40x images at MPP = 0.25
    """
    if gpu:
      # TODO: re-define device
      self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
      self.device = torch.device('cpu')
    self.save = save
    self.zoom = zoom
    self.mpp = mpp

  def convert_resolution_to_mpp(self, img_resolution=40):
    """
    Converts the solution (e.g. 20x, 40x) to microns per pixel (MPP). Calculation based on previously defined zoom&mpp during class init.
    e.g.: if 40x corresponds to 0.25 MPP, then 20x corresponds to 0.5 MPP
    """
    factor_from_defined_zoom = img_resolution / self.zoom
    new_mpp = self.mpp / factor_from_defined_zoom
    return new_mpp
    
  def run_yolo(self, img_path, mpp):
    yolo = yolo_standalone(img_path, self.device, mpp)
    args_yolo = yolo.args_init()
    nuclei_pred, patch = yolo.run_inference()

    return args_yolo


  def segment(self, img_path, cell_size = 50, img_resolution=40):
    # TODO: cell seg magic
    mpp = self.convert_resolution_to_mpp(img_resolution)
    yolo = self.run_yolo(img_path, mpp)

    return 0

      
def main():
    cell_seg_go = CSGO(gpu=False, zoom=40, mpp=0.25)
    cell_seg_go.segment('for_dev_only/TCGA-UB-AA0V-01Z-00-DX1.FB59AF14-B425-488D-94FD-E999D4057468.png')

if __name__ == '__main__':
  main()
    