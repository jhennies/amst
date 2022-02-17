#!/bin/bash

python pre_align.py \
  /media/julian/Data/datasets/20140801_hela-wt_xy5z8nm_as/140801_HPF_Gal_1_8bit/ \
  /media/julian/Data/projects/misc/amst_devel/test1_pre_align_sift_and_tm/ \
  --z_range 0 32 \
  --local_mask_range 80 -50 \
  --local_sigma 1.6 \
  --template /media/julian/Data/projects/misc/amst_devel/test_pre_align/template.tif \
  --tm_threshold 190 255 \
  --tm_sigma 0 \
  --tm_add_offset 4000 200 \
  --tm_smooth_median 8 \
  --tm_smooth_sigma 8 \
  --local_align_method sift \
  --sift_devicetype GPU
