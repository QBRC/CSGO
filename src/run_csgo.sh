# command line tool to call CSGO

# YOLO_PATH='./pretrained_weights/lung_best.float16.torchscript.pt'
# UNET_PATH='./pretrained_weights/epoch_190.pt'
img_path='../example_patches/TCGA-UB-AA0V-01Z-00-DX1.FB59AF14-B425-488D-94FD-E999D4057468.png'

cell_size=40 # 40 pixels in diameters
resolution=40 # 40x zoom on images

# most lightweight to test if CSGO is working
# python models.py --data_path ${img_path} --cell_size ${cell_size}  

# most common to capture outputs needed from the web-app
python models.py --data_path ${img_path} --cell_size ${cell_size}  --resolution ${resolution}
