class CSGO():
  def __init__(self, gpu=False, save=False):
    if gpu:
      # TODO: define device
      self.device = 0
    else:
      self.device = None
    self.save = save

  def segment(self, img_path, cell_size = 50, img_resolution=20):
    # TODO: cell seg magic
    return 0

      
    