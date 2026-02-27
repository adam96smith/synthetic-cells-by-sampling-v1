'''

Image Generator of Synthetic Motile Cells using Masks from GeneratorMask.py

'''

import numpy as np
import pickle
from tifffile import imread, imwrite
import h5py
import os
import glob
import sys
import time
import re

from synthetic_generator import texture_mask
from scipy.ndimage import median_filter, convolve, distance_transform_edt

import argparse

parser = argparse.ArgumentParser(description='Train a network.')
parser.add_argument('--input-dir', type=str, default='train_data/', help='Input Directory with 3D Masks')
parser.add_argument('--sampler-dir', type=str, default='filogen_real/', help='Directory with Sampled Fluorescence')
parser.add_argument('--output-dir', type=str, default='real_custom/', help='Target Directory for textured Images')
parser.add_argument('--aniso-factor', type=int, default=8, help='Anisotropic Factor.')
args = parser.parse_args()

assert args.input_dir[-1] == '/' # must end in /
assert args.output_dir[-1] == '/' # must end in /
assert args.sampler_dir[-1] == '/' # must end in /

# input directory
input_dir = args.input_dir
output_dir = args.output_dir
mask_files = sorted(glob.glob(input_dir+'mask_*.h5'))

# sampler directory
sampler_dir = args.sampler_dir # 'A549/ or A549_SIM/'
sampler_files = sorted(glob.glob(sampler_dir+'*.pkl'))

sampler_inds = np.random.choice(np.arange(len(sampler_files)), len(mask_files))

generation_times = []
for i, (s, mask_file) in enumerate(zip(sampler_inds, mask_files)):
    start_time = time.time()
    if i % 50 == 0:
        print(f'Texturing Mask {i+1}/{len(mask_files)}...')
        
    # load data
    with h5py.File(mask_file,'r') as f:
        dataset1 = f['data']
        mask = dataset1[...]

        # dataset2 = f['cell']
        # cell = dataset2[...]

    # try to match the sampler to the same real image
    try:
        metadata_filename = input_dir+'metadata_'+mask_file[-8:-3]+'.pkl'
        print(metadata_filename)
        with open(metadata_filename, 'rb') as f:
            metadata = pickle.load(f)
    
        cell_id = metadata['cell']
        timepoint = metadata['t']
    
        sampler_file = f'{sampler_dir}{cell_id}_t_{timepoint}.pkl'
        print(sampler_file)
        with open(sampler_file, 'rb') as f:
            sampler = pickle.load(f)
    except:
        with open(sampler_files[s], 'rb') as f:
            sampler = pickle.load(f)


    ## parameters
    
    # distmap blur (keep low to preserve image features, improves spatial continuity from mask edge)
    distmap_blur = True
    distmap_sig = 2
    
    # distmap blur (improves texture of fluorescence)
    gaussian_blur = False
    gaussian_sig = .5
    
    # rescatter parameter
    rescatter_fraction = .2

    # dist_map = distance_transform_edt(~mask[0], sampling = (32,1,1))
    # dist_map -= distance_transform_edt(mask[0], sampling = (32,1,1))
    # dist_map += 2
    dist_map = None

    image = texture_mask(mask[0], sampler, dist_map=dist_map, aniso_factor=args.aniso_factor, focal_on=False, 
                         distmap_blur=distmap_blur, distmap_sig=distmap_sig, 
                         gaussian_blur=gaussian_blur, gaussian_sig=gaussian_sig, 
                         resample_fraction=rescatter_fraction)

  
    file_index = re.findall(r'\d+', mask_file)[-2]

    os.makedirs(f'{input_dir}{output_dir}', exist_ok=True)
    with h5py.File(f'{input_dir}{output_dir}image_{file_index}.h5', 'w') as hf:
        dataset = hf.create_dataset('data', data=(image[np.newaxis]).astype(np.float32))

    # Calculate time taken for this iteration 
    iteration_time = time.time() - start_time 
    generation_times.append(iteration_time) 
    
    if i % 50 == 0:
        # Calculate the average time taken so far 
        avg_time_per_iteration = sum(generation_times) / len(generation_times) 
        
        # Estimate remaining time 
        remaining_iterations = len(mask_files) - (i + 1) 
        estimated_time_left = avg_time_per_iteration * remaining_iterations 
        
        # Convert estimated time to hours and minutes 
        hours_left = int(estimated_time_left // 3600) 
        minutes_left = int((estimated_time_left % 3600) // 60) 
        seconds_left = int(estimated_time_left % 60) 
        
        # Output estimated time remaining 
        print(f"{i+1}/{len(mask_files)}, Average Time per Cell: {avg_time_per_iteration}s, Estimated time left - {hours_left}h {minutes_left}m {seconds_left}s")