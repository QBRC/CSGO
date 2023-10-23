from hd_yolo import yolo_standalone
import argparse
import torch
from skimage.measure import label
from torch_models import UNet, SoftDiceLoss
import os
import numpy as np
import skimage
# import matplotlib.pyplot as plt

# find the path this file
SRC_DIR = os.path.realpath(os.path.dirname(__file__))

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

    return nuclei_pred, patch

  def to_device(self, x, device):
    """ Move objects to device.
        1). if x.to(device) is valid, directly call it.
        2). if x is a function, ignore it
        3). if x is a dict, list, tuple of objects.
            recursively send elements to device.
        Function makes a copy of all non-gpu objects of x. 
        It will skip objects already stored on gpu. 
    """
    try:
        return x.to(device)
    except:
        if not callable(x):
            if isinstance(x, dict):
                for k, v in x.items():
                    x[k] = self.to_device(v, device)
            else:
                x = type(x)([self.to_device(v, device) for v in x])
    return x
  
  def unet_init(self):
    '''
    Initializes the UNet model and load the model weights
    '''
    model = UNet(1, scale_factor=1.0, resize_mode='bilinear')
    model.to(self.device)
    criterion = SoftDiceLoss() 
    criterion = self.to_device(criterion, self.device)
    
    # load pretrained weights
    weight_path = os.path.join(SRC_DIR, 'for_dev_only/pretrained_weights/epoch_190.pt')
    model.load_state_dict(torch.load(weight_path, map_location=self.device))

    self.model = model

  def predict_membrane_from_patch(self, img, patch_mpp=0.25, model_mpp=0.5):
    """
    Performs membrane segmentation after resizing the original image
    @param img: the H&E stain image
           patch_mpp: the MPP of the image. Used to calculate the model input size automatically.

    @return output: membrane prediction
    """

    # resizing image to desired dimension
    # totensor Converts numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)
    # ref: https://pytorch.org/vision/main/generated/torchvision.transforms.ToTensor.html
    # image = ToTensor()(img.copy()).type(torch.float)

    # shape has value (W, H)
    original_img_shape_tuple = img.shape[0:2]
    original_img_shape = np.array(original_img_shape_tuple)

    # size of image that suits model MPP
    new_input_size = original_img_shape * patch_mpp / model_mpp
    new_input_size = new_input_size.astype(int) # convert to integer

    # resize the image
    resized_image = skimage.transform.resize(image=img,
                                             output_shape=(new_input_size[0], new_input_size[1], 3),
                                             preserve_range=True)

    # UNet architecture recommends input sizes divisible by 32
    # find the next input size fits UNet architecture
    next_multiple_of_32 = np.ceil(new_input_size / 32) * 32
    next_multiple_of_32 = next_multiple_of_32.astype(int)

    # creates a new image with the MPP-compatible image fitted to top left
    # mirror padding instead of pad with 0s
    # (a,b) means how much a to pad before the edge, and how much b to pad after the edge
    # each tuple indicates the corresponding dimension
    # ref: https://numpy.org/doc/stable/reference/generated/numpy.pad.html
    padding_specs = (
        (0, next_multiple_of_32[0]-new_input_size[0]),
        (0, next_multiple_of_32[1]-new_input_size[1]),
        (0, 0)
        )
    unet_input_image = np.pad(resized_image, pad_width=padding_specs, mode='symmetric')

    # convert image to tensor
    # input shape is now (C, H, W)
    unet_input_tensor = torch.tensor(np.transpose(unet_input_image, axes=(2, 0, 1)), dtype=torch.float)

    # predict membrane
    output = self.model(unet_input_tensor[None].to(self.device))[0].detach().cpu()
    output = output.numpy()[0] 

    # first convert back to resized image, get rid of the edges
    output = output[:new_input_size[0], :new_input_size[1]]

    # then convert back to original image size, keep only W x H. Predictions are single channel
    output = skimage.transform.resize(output, original_img_shape_tuple, preserve_range=True)

    return output

  def membrane_detection(self, patch, patch_mpp, model_mpp=0.5):
    # add a class attribute model
    self.unet_init()

    # predicts membrane
    _membrane = self.predict_membrane_from_patch(patch, patch_mpp=patch_mpp, model_mpp=model_mpp)
    membrane_mask = _membrane * 255 # convert from [0,1] to [0,255]

    return membrane_mask

  def segment(self, img_path, cell_size = 50, img_resolution=40):
    ## Nuclei segmentation with HD-Yolo ##
    patch_mpp = self.convert_resolution_to_mpp(img_resolution)
    nuclei_pred, patch = self.run_yolo(img_path, patch_mpp)

    # only want the alpha layer
    nuclei_alpha_layer = nuclei_pred[:,:,3].copy()
    
    # label each predicted nucleus with distinct numbering
    nuclei_mask = label(nuclei_alpha_layer, background=0)

    ## Membrane segmentation 
    mm = self.membrane_detection(patch, patch_mpp)
    

    return 0

      
def main():
    cell_seg_go = CSGO(gpu=False, zoom=40, mpp=0.25)
    cell_seg_go.segment('for_dev_only/TCGA-UB-AA0V-01Z-00-DX1.FB59AF14-B425-488D-94FD-E999D4057468.png')



if __name__ == '__main__':
  main()
    