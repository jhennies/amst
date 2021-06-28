
import numpy as np
from tifffile import imread
import vigra
import skimage

import shutil

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default=None)
parser.add_argument('--output', type=str, default=None)

args = parser.parse_args()

inp = args.input
out = args.output

shutil.copy(inp, out)
