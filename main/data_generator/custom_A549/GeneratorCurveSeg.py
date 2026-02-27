'''

Curvature Segmentation of Synthetic Motile Cells using Masks from GeneratorMask.py

'''

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import numpy as np
import pickle
from tifffile import imread, imwrite
import h5py
from tqdm import tqdm
import os
import glob
import sys

if os.getcwd() not in sys.path:
    (sys.path).append(os.getcwd())
        
from scipy.ndimage import convolve, binary_dilation
from skimage.morphology import remove_small_objects

import my_elektronn3
from my_elektronn3.custom.curvature import *
import re

from utils import load_config, nearest_power_of_two

import argparse

parser = argparse.ArgumentParser(description='Train a network.')
parser.add_argument('--input-dir', type=str, required=True, 
                    help='Target Directory')
parser.add_argument('--dataset-id', type=str, required=True,
                    help='Short identifier for the dataset (e.g., H157). Used for logging, config selection, etc.')
parser.add_argument('--global-config', type=str, default=None,
                    help='Config. File for Sampler')
args = parser.parse_args()
    
# input directory
mask_files = sorted(glob.glob(args.input_dir+'mask_*.h5'))


if args.global_config is None:
    global_params = load_config(f'config/{args.dataset_id}/global_parameters.yaml')
else:
    global_params = load_config(args.global_config)

# Parameters
sampling = global_params['SAMPLING']
aniso_factor = nearest_power_of_two(sampling[0]/sampling[1])

for i in tqdm(range(len(mask_files)), desc='Calculating Curvature in Synthetic Masks'):

    mask_file = mask_files[i]
        
    # load data
    metadata = {}
    with h5py.File(mask_file,'r') as f:
        dataset = f['data']
        mask = dataset[...]

    # segment the image based on curvature
    curvature_array = curvature_approximation_3d(mask[0], r=15, aniso_factor=aniso_factor, planar=False)

    # threshold curvature array
    curvature_segmentation = curvature_array > .7
    kern = circleKern_aniso(15, aniso_factor=aniso_factor)

    # localise the regions of high positive curvature
    curvature_segmentation = remove_small_objects(curvature_segmentation, 5)
    curvature_segmentation = binary_dilation(curvature_segmentation, kern)
    
    file_index = re.findall(r'\d+', mask_file)[-2]

    with h5py.File(f'{args.input_dir}curv_seg_{file_index}.h5', 'w') as hf:
        dataset = hf.create_dataset('data', data=(curvature_segmentation[np.newaxis]).astype(bool))