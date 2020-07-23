
# Convenience scripts

## smooth_displace.py

Use this function to perform displacements obtained by Fiji's template matching plugin.

    $ cd /path/to/amst/scripts/
    $ conda activate amst-env
    (amst-env)$ python smooth_displace.py --target_folder "path" --source_folder "path" --displacement_file "file.csv" [optional_parameters]

### parameters and defaults

    --target_folder "/path/to/where/to/save/the/result"
    --source_folder "/path/of/input/slices"
    --displacement_file "/filepath/of/displacements.csv"
    --median 0
    --gauss 0.0
    --n_workers 8
    --pattern "*.tif"
    --suppress_x 0
    
### Tipps and tricks

To apply the displacements as they are coming from the Fiji plugin use median=0, gauss=0 and suppress_x=0. 
Do this if TM is the first alignment step of the pipeline and the result is good for all slices.

Normally it is beneficial to run a SIFT step before alignment with TM. In this setting the image slices are already in 
place with respect to the adjacent ones (only drifts over a long range remain). 
Now it is better to smooth the TM displacements before application (e.g. median=8 and gauss=8). The advantage is that 
the correspondence of image features remain defined by the initial SIFT step whereas TM corrects for longer-range 
drifts.
Additionally, the TM plugin sometimes fails on a couple of individual slices which then does not have any effect on the 
final alignment.
Should there be larger regions where the TM struggles (e.g. when the markings went out of the ROI at image acquisition) 
the smoothing can be increased. 

In cases where only the sample surface can be used for TM to correct for a y-drift, TM can be performed selectivelly on 
on the y-axis (suppress_x=1)
