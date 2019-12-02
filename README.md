# AMST
Alignment to Median Smoothed Template for FIB-SEM data

## Installation of AMST 

### Download the source code

Clone this repository to a folder of your choice.

Create parent folder, for example:

    cd ~ 
    mkdir src
    
Navigate to this folder:

    cd ~/src
    
Clone this repository

    git clone https://github.com/jhennies/amst.git

The AMST package will now reside in /home/user/src/amst. For convenience the following instructions assume this location. 
In case you cloned the git elsewhere adapt the respective instructions accordingly.


## Installing Miniconda or Anaconda 
 
 Note: if you already have a conda python installation, jump to the next step and set up a new conda environment.

Download miniconda (lightweight version of Anaconda) from https://docs.conda.io/en/latest/miniconda.html
for python3.7 and install it to the folder of your choice.

Alternatively, if you are planning on using python later use https://www.anaconda.com/distribution/

### Set up the conda environment

#### With environment file 

Open a terminal/command line and navigate to the AMST package:

    cd /home/user/src/amst 

For Windows, installing the wheel will install the dependencies. Then jump to create environment manually.

For Linux, type:

    conda env create --file amst_env_linux.yml


#### Creating environment manually

Open a terminal/command line and follow the commands below.

Create an new environment:

    conda create --name amst_env python=3.6

#### Activate your environment:

    conda activate amst_env
    
##  Installation on Windows

- Starting from the console where you activated your amst_env in conda move to the directory you downloaded amst, then to the directory amst_win 
- Execute: 
  
    pip install vigranumpy-(press tab to autocomplete)
- Execute:

    pip install amst_bin_win-(press tab to autocomplete)

    If pyopencl gives problems (you will see error messages with pyopencl involved), execute:
        
        pip install pyopencl-(press tab to autocomplete) 
    
    Then again:
        pip install amst_bin_win-(press tab to autocomplete)

    
- Using a text editor (Notepad,...), open example_usage.py and replace the directories marked as __raw__, __aligned__ and __results__
- Execute :
    python example_usage.py

If everything went well, it will start.

## Installation on Linux

From the command line install required packages:

    conda install numpy
    conda install -c conda-forge tifffile
    conda install scikit-image
    conda install -c conda-forge vigra
    pip install pyelastix
    conda install -c conda-forge silx[full]
    conda install -c conda-forge pyopencl

Additionally, check the potential issues specified below. 


### Installation of Elastix


Extract the downloaded archive to a folder of your choice (/path/to/elastix)

Add the following to the .bashrc:

    export PATH=/path/to/elastix/bin:$PATH
    export LD LIBRARY PATH=/path/to/elastix/lib:$LD LIBRARY PATH
    
Replace '/path/to/elastix/' by the correct folder where elastix was imported to and which contains the bin and lib folders.
    
Likewise, for Windows add

    /path/to/elastix/bin
    /path/to/elastix/lib
    
to the environment variables.

Calling elastix from command line should now work, e.g.:

    $ elastix --help
    
Please also refer to the elastix documentation manual that can be downloaded here: http://elastix.isi.uu.nl


## Usage (both Windows and Linux)

An example usage can be found in example_usage.py showing the basic functionalities of AMST.
To run the script, download the example data and adapt the script according to the data location in the file system.
Open a command line and create a new folder for experiment scripts

For example:

    mkdir ~/src/amst_experiments
    cd amst_experiments
    
Copy the example script to the new folder

    cp ~/src/amst/example_usage.py my_first_amst_experiment.py
    
Adapt the script to specify the locations of the raw data, the pre-aligned data and a target folder. 
The parent folder of the target folder has to exist in your file system. If not, create it

    mkdir /path/to/target/folder 
        
Acivate the conda environment

    conda activate amst_env
    
Run the script

    python my_first_amst_experiment.py 


Additionally, check below "Known errors and issues" in case of any potential issues. 

## Parameters (OPTIONAL)

### Main parameters

The main parameters are supplied as arguments to the amst_align() function.

    amst_align(
    
Specify where to load data and save the results
        
        # Raw data
        raw_folder,
           
        # The pre-aligned data       
        pre_alignment_folder,   
        
        # Where results are saved; This folder will be created if it does not exist
        # However, the parent folder has to exist, we purposely avoided recursive folder creation
        target_folder,
                                
Settings of the amst algorithm
        
        # radius of the median smoothing surrounding
        median_radius=7,        
        
        # Parameters for the affine transformation step using Elastix; see below for more details
        elastix_params=optimized_elastix_params(),
        
        # Use SIFT to get the raw data close to the template
        sift_pre_align=True,     
        
        # Pre-smooth data before running the SIFT
        sift_sigma=1.6,   
        
        # Downsample the data for the SIFT (for speed-up, downsampling by 2 should not compromize the final result    
        sift_downsample=(2, 2),   

Computational settings
        
        # Number of CPU cores allocated
        n_workers=8,
        
        # Number of threads for the SIFT step (must be 1 if run on the GPU)
        n_workers_sift=1, 
        
        # Run the SIFT on 'GPU' or 'CPU'
        sift_devicetype='GPU',
        
Settings for debug and testing

        # Select a subset of the data for alignment (good for parameter testing)
        # To align only the first 100 slices of a dataset use
        # compute_range=np.s_[:100]
        # Note: for this to work you have to import numpy as np
        compute_range=np.s_[:]
        
        # Set to True for a more detailed console output
        verbose=False,
        
        # Set to True to also write the median smoothed template and the results of the SIFT step to disk
        # Two folders will be created within the specified target directory that contain this data ('refs' and 'sift').
        write_intermediates=False
        
    )

To obtain the defaults above, you can use the default_amst_params() function from amst_main.py which returns a dictionary to enable the following usage:

    params = default_amst_params()
    
    amst_align(
        raw_folder,
        pre_alignment_folder,
        target_folder,
        **params
    )

To modify parameters we recommend to fetch the defaults and adapt as desired, like so:

    params = default_amst_params()
    params['n_workers'] = 12
    
    amst_align(
        raw_folder,
        pre_alignment_folder,
        target_folder,
        **params
    )

### Elastix parameters

We tested and optimized Elastix parameter settings specifically for the use of AMST. 
The basis for our optimized Elastix parameters are the default Elastix parameters for affine transformations which can be found here:

http://elastix.bigr.nl/wiki/images/c/c5/Parameters_Affine.txt

These default parameter settings can be obtained as dictionary by running: 
    
    from amst_main import default_elastix_params
    
    elastix_defaults = default_elastix_params() 

The optimized parameter settings can be obtained by:

    from amst_main import optimized_elastix_params
    
    elastix_optimized = optimized_elastix_params()
    
The optimized parameter set is implemented in the default_amst_params() (see above).

The changes we introduced to the default settings are:

    # For speed-up we compromise one resolution level
    NumberOfResolutions=3,  # default=4
    
    # Still, it make sense to start down-sampling by 8 and end with no sampling
    ImagePyramidSchedule=[8, 8, 3, 3, 1, 1],  # default=(8, 8, 4, 4, 2, 2, 1, 1)
    
    # A slight speed-up, while still maintaining quality
    MaximumNumberOfIterations=200,  # default=250
    
    # For some reason turning this off really improves the result
    AutomaticScalesEstimation=False,  # default=True
    
    # Increased step length for low resolution iterations makes it converge faster (enables smaller number of
    # resolutions and iterations, i.e. speed-up of computation)
    MaximumStepLength=[4, 2, 1],  # default=1.
    
    # Similar to the default parameter "Random", a subset of locations is selected randomly. However, subpixel
    # locations are possible in this setting. Affects alignment quality
    ImageSampler='RandomCoordinate'  # default='Random'

To modify Elastix parameters for AMST we recommend to fetch AMST defaults and then modify as desired:

    params = default_amst_params()
    params['elastix_params']['MaximumNumberOfIterations'] = 500
    
    amst_align(
        raw_folder,
        pre_alignment_folder,
        target_folder,
        **params
    )
    
For details on the Elastix parameters, please also refer to the Elastix manual available here:

http://elastix.isi.uu.nl/download/elastix-5.0.0-manual.pdf

## Known errors and issues

### 1. pyopencl._cl.LogicError: clGetPlatformIDs failed: PLATFORM_NOT_FOUND_KHR

OpenCL cannot find the proper drivers. This affects the SIFT alignment step which gets the raw close to the template before running Elastix.

To get the GPU working on Linux, copy the graphics vendors (e.g. Nvidia.icd) from 
    
    /etc/OpenCL/vendors 
    
to 

    /path/to/miniconda3/envs/amst_env/etc/OpenCL/
    
Create the OpenCL folder if necessary. 

Alternatively, to get the SIFT running at least on the CPU:

In the conda evironment install:

    conda install -c conda-forge pocl

### 2. self.ctx = ocl.create_context(devicetype=devicetype, AttributeError: 'NoneType' object has no attribute 'create_context'

This has, so far, only occured on Windows machines. Similar to 1., OpenCL cannot be instantiated. We were able to fix this by renaming the OpenCL.dll in C:\Windows\System32\ to, e.g., OpenCL.dll.bak.
    
### 3. RuntimeError: An error occured during registration: [Errno 2] No such file or directory: '/tmp/pyelastix/id_25994_140493512837272/result.0.mhd'

This is a reported bug in the pyelastix package (which does not affect Windows, apparently). To fix it, do the following:

change /path/to/miniconda3/envs/amst_env_devel/lib/python3.6/site-packages/pyelastix.py line 304 

        p = subprocess.Popen(cmd, shell=True,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                         
to 

        p = subprocess.Popen(cmd, shell=False,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

Also see https://github.com/almarklein/pyelastix/pull/8
    
### 4. Result data seems all-zero (all images are black) 

It seems to be some debug code left in the pyelastix package.
Check line 558 in /path/to/miniconda3/envs/amst_env_devel/lib/python3.6/site-packages/pyelastix.py. If it is 

    im = im* (1.0/3000)
    
delete the line.


#### 4. Problems with module 'Module not found error'
In some occasions is not possible to download a package properly from conda or pip. If that is the case, download the corresponding
wheel from a different repository. For example, you can use the wheels from Christoph Golke page, for example, for Vigra:
https://www.lfd.uci.edu/~gohlke/pythonlibs/#vigra

Download the .whl file and then install using pip:
   pip install dowloaded_package.whl