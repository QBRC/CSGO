class CSGO():
  def __init__(self, gpu=False, save=False, zoom=40, mpp=0.25):
    """
    Define model level info.
    zoom: 1st data point to convert resolution and mpp
    mpp: 2nd data point to convert resolution and mpp
    standard equipment places 40x images at MPP = 0.25
    """
    if gpu:
      # TODO: define device
      self.device = 0
    else:
      self.device = None
    self.save = save
    self.zoom = zoom
    self.mpp = mpp

  def convert_resolution_to_mpp(self, img_resolution):
    """
    Converts the solution (e.g. 20x, 40x) to microns per pixel (MPP). Calculation based on previously defined zoom&mpp during class init.
    e.g.: if 40x corresponds to 0.25 MPP, then 20x corresponds to 0.5 MPP
    """
    factor_from_defined_zoom = img_resolution / self.zoom
    new_mpp = self.mpp / factor_from_defined_zoom
    return new_mpp
    

  def segment(self, img_path, cell_size = 50, img_resolution=20):
    # TODO: cell seg magic
    return 0

      
    