# AMST
Alignment to Median Smoothed Template for FIB-SEM data

## Installation of AMST

### Installation of Elastix

Please also refer to the exlastix documentation manual that can be downloaded here: http://elastix.isi.uu.nl/doxygen/index.html

Extract the downloaded archive to a folder of your choice

For linux add the following to the .bashrc:

    export PATH=folder/bin:$PATH
    export LD LIBRARY PATH=folder/lib:$LD LIBRARY PATH
    
Calling elastix from command line should now work, e.g.:

    $ elastix --help

### Set up the conda environment

Create the environment and install the following packages like so:

    conda create --name amst_env python=3.6
    conda install numpy
    conda install -c conda-forge tifffile
    conda install -c conda-forge vigra
    conda install scikit-image
    pip install pyelastix
    conda install -c conda-forge silx[full]

## Usage

An example usage can be found in example_with_sift_advanced_workflow.py showing the basic functionalities of AMST.

## Possible issues

If you encounter an error like this

    RuntimeError: An error occured during registration: [Errno 2] No such file or directory: '/tmp/pyelastix/id_25994_140493512837272/result.0.mhd'
    
change pyelastix.py line 304 

        p = subprocess.Popen(cmd, shell=True,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                         
to 

        p = subprocess.Popen(cmd, shell=False,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

Also see https://github.com/almarklein/pyelastix/pull/8
                   
If your result data seems all-zero check line 558 in pyelastix.py. If it is 

    im = im* (1.0/3000)
    
delete it.
