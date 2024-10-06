# CSGO

## Introduction
Cell Segmentation with Globally Optimized boundaries (CSGO) is a whole-cell segmentation pipeline tailored specifically to the H&E domain. CSGO brings four notable contributions: 1) a U-Net algorithm to detect cell membranes with various intensities, 2) integration of HD-Yolo nuclei detection with the U-Net membrane detection, 3) superior performance and robustness: CSGO consistently outperforms the state-of-the-art cell segmentation algorithm across a range of cancer types, all while requiring fewer manual annotations. Furthermore, our pipeline demonstrates remarkable robustness even in scenarios with significant variations in individual cell sizes.

If you would like to try CSGO without going through the installation process below, a web application is available at https://ai.swmed.edu/projects/csgo/index.php

## Citation
If you used the tools within this repository, please cite us at:
TBD

## Install package Using Conda

```
git clone git@github.com:QBRC/CSGO.git
cd CSGO/
conda env create -f environment.yml 
conda activate cell-seg-go
```
## Preparing to use CSGO
Before running CSGO, check if `src/pretrained_weights` contain file `epoch_190.pt`. This is the pretrained UNet weight. You would also need the pretrained HD-Yolo weights. HD-Yolo weights are too large to be stored on GitHub. Please download from here:
<ul>
   <li> <a href="https://drive.google.com/file/d/131RQwmrQeonwuLr46L06gWZ8Jv60opSt/view?usp=share_link">HD-Yolo weights</a> </li>
</ul>

Move the downloaded HD-Yolo weights to `src/pretrained_weights`

```
mv /path/to/lung_best.float16.torchscript.pt src/pretrained_weights
```

## How to use CSGO in command line
`src/run_csgo.sh` provides few examples on how you can run CSGO. The most straightforward way is to specify the patch location only:
```
cd src/
python models.py --data_path /path/to/your/image
```

If your equipment has resolution of MPP=0.25 at 40x, but the patch itself is in a different resolution, you can specify the resolution as well as the new cell size:
```
python models.py --data_path ${img_path} --cell_size ${cell_size}  --resolution ${resolution}
```

If your equipment does not have the standard resolution of MPP=0.25 at 40x, you can specify the mpp at corresponding MPP by:
```
python models.py --data_path ${img_path} --cell_size ${cell_size} --zoom_for_mpp ${zoom_for_mpp} --mpp_for_zoom ${mpp_for_zoom}  --resolution ${resolution}
```

Below is the helper information for all available args:

```
usage: Whole-cell segmentation with CSGO. [-h] --data_path DATA_PATH [--yolo_path YOLO_PATH] [--unet_path UNET_PATH]
                                          [--output_dir OUTPUT_DIR] [--save SAVE] [--gpu GPU] [--zoom_for_mpp ZOOM_FOR_MPP]
                                          [--mpp_for_zoom MPP_FOR_ZOOM] [--resolution RESOLUTION] [--cell_size CELL_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        Input data filename.
  --yolo_path YOLO_PATH
                        HD-Yolo model path, torch jit model.
  --unet_path UNET_PATH
                        UNet model path, torch jit model.
  --output_dir OUTPUT_DIR
                        Output folder.
  --save SAVE           Option to save the model outputs
  --gpu GPU             Boolean. Run on gpu if true else on cpu.
  --zoom_for_mpp ZOOM_FOR_MPP
                        Zoom: 1st data point to convert resolution and mpp. Standard is 40x at 0.25mpp. This is
                        microscope/equipment specific.
  --mpp_for_zoom MPP_FOR_ZOOM
                        MPP: 2nd data point to convert resolution and mpp. Standard is 40x at 0.25mpp. This is microscope/equipment
                        specific
  --resolution RESOLUTION
                        Input patch resolution.
  --cell_size CELL_SIZE
                        Default cell size (diameter), measured in pixels.
```


## How to use CSGO in Jupyter Notebooks
Please find a notebook tutorial of running CSGO with example images here: [csgo_tutorial.ipynb](notebooks/csgo_tutorial.ipynb).
