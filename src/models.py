from hd_yolo import yolo_standalone
import argparse
import torch
from skimage.measure import label
from torch_models import UNet, SoftDiceLoss
import os
import numpy as np
import skimage
from scipy import ndimage as ndi
from scipy.spatial.distance import cdist
from skimage.segmentation import watershed
import matplotlib.pyplot as plt
from skimage.color import label2rgb

# find the path this file
SRC_DIR = os.path.realpath(os.path.dirname(__file__))

class CSGO():
  def __init__(self, yolo_path=None, unet_path=None, gpu=False, save=False, output_dir=None, zoom=40, mpp=0.25):
    """
    Define high-level attributes for CSGO (Cell Segmentation with Globally Optimized boundaries).

    Parameters
    ----------
    yolo_path : str
        Path of the HD-Yolo model pretrained weight. HD-Yolo predicts nuclei
    unet_path : str
        Path of the U-Net model pretrained weight. U-Net predicts membrane
    gpu : bool, default False
        Whether to use GPU to evaluate the models
    save : bool, default False
        Whether to save CSGO outputs (see more detailed under `Notes`)
    output_dir : str
        The path to save CSGO outputs 
    zoom : int, default 40
        The first data point needed to convert resolution and mpp
    mpp : int, default 0.25
        The second data point needed to convert resolution and mpp

    Notes
    -----
    If `save` is enabled, 1) the cell segmentation result, where each pixel is assigned to a cell, and 2) an image showing the process CSGO will be written to `output_dir`.
    Standard imaging equipment places 40x zoomed images at MPP = 0.25
    """
    if gpu:
      # TODO: re-define device
      self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
      self.device = torch.device('cpu')
    self.save = save
    self.output_dir = output_dir
    self.zoom = zoom
    self.mpp = mpp
    self.yolo_path = yolo_path
    self.unet_path = unet_path

  def convert_resolution_to_mpp(self, img_resolution=40):
    """
    Converts the solution (e.g. 20x, 40x) to microns per pixel (MPP). Calculation based on previously defined zoom&mpp during class init.
    e.g.: if 40x corresponds to 0.25 MPP, then 20x corresponds to 0.5 MPP
    """
    factor_from_defined_zoom = img_resolution / self.zoom
    new_mpp = self.mpp / factor_from_defined_zoom
    return new_mpp
    
  def run_yolo(self, img_path, mpp):
    yolo = yolo_standalone(img_path, self.device, mpp, self.yolo_path)
    args_yolo = yolo.args_init()
    nuclei_pred, patch = yolo.run_inference()

    return nuclei_pred, patch

  def to_device(self, x, device):
    """ Move objects to device.
        Option 2 and 3 not expected in this app.
        1). if x.to(device) is valid, directly call it.
        2). if x is a function, ignore it
        3). if x is a dict, list, tuple of objects.
            recursively send elements to device.
        Function makes a copy of all non-gpu objects of x. 
        It will skip objects already stored on gpu. 
    """
    return x.to(device)
  
  def unet_init(self):
    '''
    Initializes the UNet model and load the model weights
    '''
    model = UNet(1, scale_factor=1.0, resize_mode='bilinear')
    model.to(self.device)
    criterion = SoftDiceLoss() 
    criterion = self.to_device(criterion, self.device)
    
    # load pretrained weights
    model.load_state_dict(torch.load(self.unet_path, map_location=self.device))

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

  def find_missed_nuclei(self, ndi_distance, nucleus_label, cell_size):
      """
      Given transformed distance from ndi, find the local minima and mark them as new nucleus
      @param ndi_distance: topology map of membrane and nucleus computed from ndi.distance_transform_edt()
            nucleus_label: nucleus prediction results from HD-Staining
            cell_size: the average diameter of the cell. Used to calculate minimum separation between two nucleus, and the size of the artifical nucleus
                for minimum searation, it is used in 1)peak_local_max(), and 2)the local minima will be only be a new nucleus if at least this disntance away from HD-Staining label
      
      @return nucleus_label: nucleus labels found by both HD-Staining and minimum local distances
      """

      # determines minimum separation
      min_nucleus_distance = int(cell_size/2)
      
      # negative here to find the maximum
      # ref: https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.peak_local_max
      local_min = skimage.feature.peak_local_max(-ndi_distance, min_distance=min_nucleus_distance, exclude_border=False)

      # preserve the shape as HD-Staining predicted nucleus
      local_min_mask = np.zeros(nucleus_label.shape, dtype=bool)

      # Set the elements corresponding to the coordinates in `local_min` to True
      local_min_mask[local_min[:, 0], local_min[:, 1]] = True

      # local_min_mask_dilated = skimage.morphology.binary_dilation(local_min_mask, skimage.morphology.disk(60))
      local_min_mask_labeled = skimage.measure.label(local_min_mask)

      artifical_nucleus_size = int(cell_size / 3)

      for label in np.unique(local_min_mask_labeled):
        if label == 0: continue # skip the background

        # build individually labeled minima masks
        ind_minima_labeled = local_min_mask_labeled == label

        # calculate the Euclidean distance between each pixel in the image (i.e. HD-Staining nucelus prediction) and each pixel in the label
        # label in this case only has True or False
        distances = cdist(np.argwhere(nucleus_label), np.argwhere(ind_minima_labeled))

        # the closest other nucleus is too far to be within the same cell
        # distance between two local minima is not checked, as it was filtered in peak_local_max()
        if np.min(distances) >= min_nucleus_distance:
          # add to nuc_zero label as a new nuclei
          # idea: average nucleus size in each predicted labels
          ind_nuclei = skimage.morphology.binary_dilation(ind_minima_labeled, skimage.morphology.disk(artifical_nucleus_size))
          nucleus_label[(nucleus_label == 0) & (ind_nuclei == 1)] = np.unique(nucleus_label)[-1] + 1 # mark the membrane as another color on "nucleus" labels

      return nucleus_label
  
  def watershed(self, nuclei_mask, membrane_mask, cell_size):
    """
    performs watershed using predicted nucleus (HD-Yolo) and membranes (UNet).
    """

    # negative membrane for distrance transform
    new_mm  = 1-membrane_mask/255

    # picking a threshhold to create binary label
    new_mm[new_mm > 0.5] = 1
    new_mm[new_mm <= 0.5] = 0  

    # fill possible small holes in predictions
    # function always outputs boolean, therefore membrane predictions are first filled, then used to guide watershed 
    # ref: https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.remove_small_holes
    # new_mm = remove_small_holes(new_mm.astype(bool), area_threshold=100)

    distance_mm=-ndi.distance_transform_edt(new_mm)

    # in watershed, label=0 means the pixel (i.e. those that were not predicted as nucleus) is not a marker. Ref: https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.watershed
    # transforming distance based on nucleus. Further away from membrane, the lower the water levels are --> nucleus = basin
    distance_nuc=ndi.distance_transform_edt(nuclei_mask==0)  # nuclei masks have unique values. 0 indicates non-nuclei
    
    # combine membrane distance and nuclei distance
    distance = distance_mm + distance_nuc/4

    # find nuclei missed by HD-Yolo
    nuc_zero = self.find_missed_nuclei(distance, nuclei_mask, cell_size)

    # in watershed, label=0 means not the pixel is not a marker. Ref: https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.watershed
    markers = nuc_zero.copy() 

    # markers indicate where the basins are
    # labels will follow the same order as markers, i.e. largest value in marker is the membrane, and that of in label will be membrane.
    watershed_res = watershed(distance, markers=markers)

    return watershed_res



  def segment(self, img_path, cell_size = 50, img_resolution=40):
    ## Nuclei segmentation with HD-Yolo ##
    patch_mpp = self.convert_resolution_to_mpp(img_resolution)
    nuclei_pred, patch = self.run_yolo(img_path, patch_mpp)

    # only want the alpha layer
    nuclei_alpha_layer = nuclei_pred[:,:,3].copy()
    
    # label each predicted nucleus with distinct numbering
    nuclei_mask = label(nuclei_alpha_layer, background=0)

    ## Membrane segmentation with Unet ##
    membrane_mask = self.membrane_detection(patch, patch_mpp)

    ## Watershed with nuclei and membrane detection ##
    # raw segmentation result (i.e. each pixel assigned to a cell)
    res_cell_seg = self.watershed(nuclei_mask, membrane_mask, cell_size)

    # save the results
    if self.save:

      # set color palette
      cmap_set3   = plt.get_cmap("Set3")  
      cmap_tab20c = plt.get_cmap("tab20c")  
      color_dict = [cmap_tab20c.colors[_] for _ in range(len(cmap_tab20c.colors))] + \
        [cmap_set3.colors[_] for _ in range(len(cmap_set3.colors))]

      # this output is for web app
      fig, axs = plt.subplots(1,4, figsize=(18, 6))
      axs[0].imshow(patch), axs[0].set_title('Uploaded H&E patch')
      axs[1].imshow(label2rgb(nuclei_mask, colors=color_dict)), axs[1].set_title('Nuclei mask from HD-Yolo')
      axs[2].imshow(membrane_mask), axs[2].set_title('Membrane mask from UNet')
      axs[3].imshow(label2rgb(res_cell_seg, colors=color_dict)), axs[3].set_title('CSGO whole-cell segmentation')
      plt.savefig(os.path.join(self.output_dir, 'pipeline_view.png'), dpi=300)

      # this output is for user download
      res_cell_seg = res_cell_seg.astype(np.uint8)
      skimage.io.imsave(os.path.join(self.output_dir, 'CSGO_whole_cell_seg.png'), res_cell_seg)

    return res_cell_seg


    

      
def main():
    parser = argparse.ArgumentParser('Whole-cell segmentation with CSGO.', add_help=True)
    
    # app/model args
    parser.add_argument('--data_path', required=True, type=str, help="Input data filename.")
    parser.add_argument('--yolo_path', default='pretrained_weights/lung_best.float16.torchscript.pt', type=str, help="HD-Yolo model path, torch jit model." )
    parser.add_argument('--unet_path', default='pretrained_weights/epoch_190.pt', type=str, help="UNet model path, torch jit model." )
    parser.add_argument('--output_dir', default='patch_results', type=str, help="Output folder.")
    parser.add_argument('--save', default=True, type=bool, help='Option to save the model outputs')
    
    # patch args
    parser.add_argument('--gpu', default=False, type=bool, help='Boolean. Run on gpu if true else on cpu.')
    parser.add_argument('--zoom_for_mpp', default=40, type=int, help='Zoom: 1st data point to convert resolution and mpp. Standard is 40x at 0.25mpp. This is microscope/equipment specific.')
    parser.add_argument('--mpp_for_zoom', default=0.25, type=float, help='MPP: 2nd data point to convert resolution and mpp. Standard is 40x at 0.25mpp. This is microscope/equipment specific')
    parser.add_argument('--resolution', default=40, type=float, help='Input patch resolutio.')
    parser.add_argument('--cell_size', default=50, type=int, help='Default cell size (diameter), measured in pixels.')
       
    args = parser.parse_args()

    # initialize
    if not os.path.exists(args.output_dir):
      os.makedirs(args.output_dir)

    # perform segmentation and save
    cell_seg_go = CSGO(yolo_path=args.yolo_path, unet_path=args.unet_path, gpu=args.gpu, save=args.save, output_dir=args.output_dir, zoom=args.zoom_for_mpp, mpp=args.mpp_for_zoom)
    cell_seg_go.segment(args.data_path, args.cell_size, args.resolution)


if __name__ == '__main__':
  main()
    