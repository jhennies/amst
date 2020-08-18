
# Convenience scripts

## compress.py

Use this function to change the compression level of a folder of tif slices. 

    $ cd /path/to/amst/scripts/
    $ conda activate amst-env
    (amst-env)$ python compress.py --target_folder /path --source_folder /path [optional parameters]

### parameters and defaults

#### mandatory

    --target_folder /path/to/where/to/save/the/result
    --source_folder /path/of/input/slices
    
Explanations:
 - target_folder: A folder where to save the result. If not existing, this folder will be created, however the 
 parent folder has to exist.
 - source_folder: Folder that contains the input slices
    
#### optional

    --pattern *.tif
    --z_range 0
    --compression 9
    --n_workers 8
    --verbose 0
    
Explanations:
 - pattern: File pattern of the input slices. The default *.tif is usually fine for a folder of tif slices.
 - z_range: Defines a subset of the input tif slices
 - compression: Defines the compression level from no compression (0) to maximum compression (9)
 - n_workers: Number of CPU cores used for calculation
 - verbose: Level of console output
 
 
## run_amst.py

Runs the very basic version of AMST. For a more custom setting refer to the example_usage.py script. 

    $ cd /path/to/amst/scripts/
    $ conda activate amst-env
    (amst-env)$ python run_amst.py --raw_folder /path --pre_alignment_folder /path --target_folder /path [optional parameters]

### parameters and defaults

#### mandatory

    --raw_folder /path/to/raw/data
    --pre_alignment_folder /path/to/realignment
    --target_folder /path/where/to/save/result
    
Explanations:
 - raw_folder: Where the raw data resides
 - pre_alignment_folder: Where the pre-alignment resides
 - target_folder: A folder where to save the result. If not existing, this folder will be created, however the 
 parent folder has to exist.
    
#### optional

    --n_workers 8
    --verbose 0
    
Explanations:
 - n_workers: Number of CPU cores used for calculation
 - verbose: Level of console output

## smooth_displace.py

Use this function to perform displacements obtained by Fiji's template matching plugin.

    $ cd /path/to/amst/scripts/
    $ conda activate amst-env
    (amst-env)$ python smooth_displace.py --target_folder "path" --source_folder "path" --displacement_file "file.csv" [optional_parameters]

### parameters and defaults

#### mandatory:

    --target_folder /path/to/where/to/save/the/result
    --source_folder /path/of/input/slices
    --displacement_file /filepath/of/displacements.csv
    
Explanations:
 - target_folder: A folder where to save the result. If not existing, this folder will be created, however the 
 parent folder has to exist.
 - source_folder: Folder that contains the input slices
 - displacement_file: csv file created by saving the results table of Fiji's template matching plugin

#### optional:

    --median 0
    --gauss 0.0
    --n_workers 8
    --pattern *.tif
    --suppress_x 0
    --source_range 0
    
Explanations:
 - median: Median-smoothing of the displacements before application. Specifies the radius of this median filter
 - gauss: Gaussian smoothing of the displacements before application. Specifies the gaussian sigma
 - n_workers: Number of CPU cores used for calculation
 - pattern: File pattern of the input slices. The default *.tif is usually fine for a folder of tif slices.
 - suppress_x: If set to 1, only the y component of the displacements is applied to the slices
 - source_range: Defines a subset of the input tif slices. The size of the subset must match the number of entries in 
 the displacement file.
 
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

The source range parameter defines only the range of the tif slices in the source_folder to be used. The IDs in the 
displacement file are kept as they are. 

