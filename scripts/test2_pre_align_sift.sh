#!/bin/bash

python pre_align.py \
  /media/julian/Data/datasets/20140801_hela-wt_xy5z8nm_as/140801_HPF_Gal_1_8bit/ \
  /media/julian/Data/projects/misc/amst_devel/test2_pre_align_sift/ \
  --z_range 0 32 \
  --local_mask_range 100 0 \
  --local_sigma 1.6 \
  --local_align_method sift \
  --sift_devicetype GPU
