# AMST
Alignment to Median Smoothed Template for FIB-SEM data

## Installation of AMST

### Installation of Elastix

Please also refer to the exlastix documentation manual that can be downloaded here: http://elastix.isi.uu.nl/doxygen/index.html

Extract the downloaded archive to a folder of your choice (/path/to/elastix)

For Linux add the following to the .bashrc:

    export PATH=/path/to/elastix/bin:$PATH
    export LD LIBRARY PATH=/path/to/elastix/lib:$LD LIBRARY PATH
    
Replace '/path/to/elastix/' by the correct folder where elastix was imported to and which contains the bin and lib folders.
    
Likewise, for Windows add

    /path/to/elastix/bin
    /path/to/elastix/lib
    
to the environment variables.

Calling elastix from command line should now work, e.g.:

    $ elastix --help
    
### Installing Miniconda

Download miniconda from https://docs.conda.io/en/latest/miniconda.html
for python3.7 and install it to the folder of your choice.

### Set up the conda environment

Open a terminal/command line and follow the commands below.

Create an new environment:

    conda create --name amst_env python=3.6

Activate it:

    conda activate amst_env
    
Install required packages:

    conda install numpy
    conda install -c conda-forge tifffile
    conda install scikit-image
    conda install -c conda-forge vigra
    pip install pyelastix
    conda install -c conda-forge silx[full]
    conda install -c conda-forge pyopencl

Additionally, check the potential issues specified below. 

## Usage

An example usage can be found in example_hela_dataset.py showing the basic functionalities of AMST.
To run the script, download the example data and adapt the script according to the data location in the file system.
Open a command line and navigate to the amst_package

    cd /path/to/amst/
    
Acivate the conda environment

    conda activate amst_env
    
Run the script

    python example_hela_dataset.py 

## Known errors and issues

### 1. pyopencl._cl.LogicError: clGetPlatformIDs failed: PLATFORM_NOT_FOUND_KHR

OpenCl cannot find the proper drivers. This affects the SIFT alignment step which gets the raw close to the template before running Elastix.

To get the GPU working on Linux, copy the graphics vendors (e.g. Nvidia.icd) from 
    
    /etc/OpenCL/vendors 
    
to 

    /path/to/miniconda3/envs/amst_env/etc/OpenCL/
    
Create the OpenCL folder if necessary. 

Alternatively, to get the SIFT running at least on the CPU:

In the conda evironment install:

    conda install -c conda-forge pocl
    
    
### 2. RuntimeError: An error occured during registration: [Errno 2] No such file or directory: '/tmp/pyelastix/id_25994_140493512837272/result.0.mhd'

This is a reported bug in the pyelastix package (which does not affect Windows, apparently). To fix it, do the following:

change /path/to/miniconda3/envs/amst_env_devel/lib/python3.6/site-packages/pyelastix.py line 304 

        p = subprocess.Popen(cmd, shell=True,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                         
to 

        p = subprocess.Popen(cmd, shell=False,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

Also see https://github.com/almarklein/pyelastix/pull/8
    
### 3. Result data seems all-zero 

There seems to be some debug code left in the pyelastix package.
Check line 558 in /path/to/miniconda3/envs/amst_env_devel/lib/python3.6/site-packages/pyelastix.py. If it is 

    im = im* (1.0/3000)
    
delete it.
