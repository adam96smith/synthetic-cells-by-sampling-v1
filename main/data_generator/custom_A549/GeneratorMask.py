'''
Binary Mask Generator of Synthetic Motile Cells 

'''

import numpy as np
from skimage.draw import line_nd
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt, zoom
from skimage.morphology import remove_small_objects
import random
import os
import h5py
from tqdm import tqdm
import pickle
import sys

if os.getcwd() not in sys.path:
    (sys.path).append(os.getcwd())

import my_elektronn3
from my_elektronn3.custom.perlin_noise import *
from my_elektronn3.custom.curvature import *
from my_elektronn3.data import transforms

from utils import load_config, max_down_sample, nearest_power_of_two

import argparse

parser = argparse.ArgumentParser(description='Generate Synthetic Cell Shapes.')
parser.add_argument('--save-path', type=str, default='synthetic_train/', 
                    help='Output Directory')
parser.add_argument('--N', type=int, default=100, 
                    help='Number of Total Generated Samples.')
parser.add_argument('--dataset-id', type=str, required=True,
                    help='Short identifier for the dataset (e.g., H157). Used for logging, config selection, etc.')
parser.add_argument('--global-config', type=str, default=None,
                    help='Config. File for Sampler')
args = parser.parse_args()


def norm(arr):
    ''' Normalises array values between 0 and 1 '''
    return(arr-np.min(arr))/(np.max(arr)-np.min(arr)+1e-5)


def circleKern(N, dims=3):

    if dims==2:
        assert N % 2 == 1
    
        X, Y = np.meshgrid(np.arange(N), np.arange(N))
    
        loc = [N//2, N//2]
    
        out = np.sqrt((X-loc[0])**2 + (Y-loc[1])**2 ) < N/2
        
    elif dims==3:
        assert N % 2 == 1
    
        X, Y,Z = np.meshgrid(np.arange(N), np.arange(N), np.arange(N))
    
        loc = [N//2, N//2, N//2]
    
        out = np.sqrt((X-loc[0])**2 + (Y-loc[1])**2 + (Z-loc[2])**2) < N/2

    else: 
        raise Exception('dims must be 2 or 3')

    return out

def taper_function(r, a=.01, c=.5):
    return np.minimum(np.ones(r.shape), a*r+c)


### Main

assert args.save_path[-1] == '/'

os.makedirs(args.save_path, exist_ok=True)

if args.global_config is None:
    global_params = load_config(f'config/{args.dataset_id}/global_parameters.yaml')
else:
    global_params = load_config(args.global_config)

# Parameters
sampling = global_params['SAMPLING']
aniso_factor = nearest_power_of_two(sampling[0]/sampling[1])

zres, xres, yres = 128, 160, 208

# filopodia parameters by type
filo_params = {'thick':{'mean_len':40, 
                        'step_sig':10, 
                        'step_pert':.1, 
                        'filo_width':1,
                        'taper_prob':1, 
                        'taper_a':.02, 
                        'taper_c':.0,},               
              'thin_long':{'mean_len':30, 
                           'step_sig':1, 
                           'step_pert':.2, 
                           'filo_width':1,
                           'taper_prob':0, 
                           'taper_a':.005, 
                           'taper_c':.1,},
              'thin_short':{'mean_len':10, 
                            'step_sig':1, 
                            'step_pert':.5, 
                            'filo_width':1,
                            'taper_prob':1, 
                            'taper_a':.2, 
                            'taper_c':.0,},
              }

choice = np.concatenate( (np.zeros([args.N//3]),
                          np.ones([args.N//3]), 
                          2*np.ones([args.N - 2*(args.N//3)]) ) ) # array of 0s, 1s & 2s
random.shuffle(choice)

# transform
EPT = transforms.ElasticPerlinTransform(prob=1,grid_points=32,p=.025)

metadata = {}
for i in tqdm(range(args.N), desc='Generating Image Masks'):

    ## What phenotype
    x = choice[i]

    ## Image Parameters (from Castilla 2019)

    if x == 0: # wt
        cell_type = 'WT'
        n_thick_branches = 0
        n_long_thin_branches = 0
        n_short_thin_branches = int(1 + 3*np.random.rand())

    elif x == 1:
        cell_type = 'OE'
        n_thick_branches = 0
        n_long_thin_branches = 0
        n_short_thin_branches = int(7 + 8*np.random.rand())
        
    else: # mutant
        cell_type = 'PD'
        n_thick_branches = int(3*np.random.rand())
        n_long_thin_branches = int(2 + 2*np.random.rand())
        n_short_thin_branches = int(7 + 8*np.random.rand())

    
    cell_radius = 25
    branch_numbers = [n_thick_branches, n_long_thin_branches, n_short_thin_branches] 
    

    # sample the cell centroid
    centroid_sample_space_z = np.arange(48, zres-48)
    centroid_sample_space_x = np.arange(48, xres-48)
    centroid_sample_space_y = np.arange(48, yres-48)
    if len(centroid_sample_space_z) > 0 and len(centroid_sample_space_x) > 0 and len(centroid_sample_space_y) > 0:
        loc = np.concatenate((np.random.choice(centroid_sample_space_z, 1),
                              np.random.choice(centroid_sample_space_x, 1),
                              np.random.choice(centroid_sample_space_y, 1))).astype(int)

    # generate cell shape
    X, Z, Y = np.meshgrid(np.arange(xres), np.arange(zres), np.arange(yres))
    R = np.sqrt((Z-loc[0])**2+(X-loc[1])**2+(Y-loc[2])**2)
    
    mask = (R<cell_radius)
    _, [mask] = EPT(mask[np.newaxis], [mask])
    
    R_surf = distance_transform_edt(mask==0) # distance to cell surface

    # add filopodia branches
    branches = {}
    branch_counter = 1
    for n_branches, branch_type in zip(branch_numbers, ['thick', 'thin_long', 'thin_short']):
    
        mean_len = filo_params[branch_type]['mean_len']
        step_sig = filo_params[branch_type]['step_sig']
        step_pert = filo_params[branch_type]['step_pert']
        filo_width = filo_params[branch_type]['filo_width']
        taper_prob = filo_params[branch_type]['taper_prob']
        taper_a = filo_params[branch_type]['taper_a']
        taper_c = filo_params[branch_type]['taper_c']
    
        outline = binary_dilation(~mask) * mask
        tmp = np.vstack([Z[outline], X[outline], Y[outline]])
        sample_inds = np.random.choice(np.arange(tmp.shape[1]), n_branches)
        
        for idx in sample_inds:
            filo_loc = tmp[:,idx]
        
            # sample direction
            direction_found = True
            n = 0
        
            dz = filo_loc[0] - zres/2
            dx = filo_loc[1] - xres/2
            dy = filo_loc[2] - yres/2
            dr = np.sqrt(dz**2+dx**2+dy**2)
        
            # intialise filopodia pointing outward + perturb
            theta = np.arctan2(dy, dx) #+ np.random.normal(loc=0, scale=.5)
            # phi = np.arccos(dz/dr) #+ np.random.normal(loc=0, scale=.1)
            phi = np.pi/2 # always planar
            
            # create filopodia trajectories
            curr_loc = filo_loc
            curr_theta = theta
            curr_phi = phi
            curr_len = 0
            targ_len = np.random.normal(loc=mean_len, scale=mean_len/10)
            
            points_x = []
            points_y = []
            points_z = []
    
            edge_hit = False
            while curr_len < targ_len and not edge_hit: # stop if you get to the edge
                # sample new length and direction
                r = np.random.exponential(scale=step_sig)
                th = np.random.normal(loc=0, scale=step_pert)
                phi = np.random.normal(loc=0, scale=.1)
            
                new_z = int(curr_loc[0] + r*np.cos(phi + curr_phi))
                new_x = int(curr_loc[1] + r*np.sin(phi + curr_phi)*np.cos(th + curr_theta))
                new_y = int(curr_loc[2] + r*np.sin(phi + curr_phi)*np.sin(th + curr_theta))
            
                if 0<new_z<zres and 0<new_x<xres and 0<new_y<yres:
                    edge_hit == False
                else:
                    new_z = max(min(new_z, zres-1),0)
                    new_x = max(min(new_x, xres-1),0)
                    new_y = max(min(new_y, yres-1),0)
                    
                    edge_hit == True
                    
                # add to sequence
                points_z.append( [curr_loc[0], new_z] )
                points_x.append( [curr_loc[1], new_x] )
                points_y.append( [curr_loc[2], new_y] )
                
            
                # update current data
                curr_loc = [new_z, new_x, new_y]
                curr_theta += th
                curr_phi += phi
                curr_len += r
        
            branches[str(branch_counter)] = {'length': curr_len, 'type': branch_type, 'x':points_x,'y':points_y,'z':points_z,}
            branch_counter += 1
        
        # then add all branches
        for s in branches:
            points_z = branches[s]['z']
            points_x = branches[s]['x']
            points_y = branches[s]['y']
            
            mask_filo = np.zeros(mask.shape)
            
            for [z0,z1], [x0,x1], [y0,y1] in zip(points_z, points_x, points_y):            
                l0, l1, l2 = line_nd([z0,x0,y0],[z1,x1,y1])
                mask_filo[l0, l1, l2] = 1
    
            if np.random.rand() < taper_prob:
                filopodia = taper_function(R_surf, a=taper_a, c=taper_c) * distance_transform_edt(mask_filo==0) <= filo_width
            else:
                filopodia = (distance_transform_edt(mask_filo==0) <= filo_width)
                
            mask = np.maximum(mask, filopodia)

    
    # mask = binary_erosion(mask)
    
    # aniso-factor
    mask = max_down_sample(mask, aniso_factor//2) # already downsampled by 2x2x2
    
    # upscale x-y 
    mask = zoom(mask, (1,2,2), order=1)

    # fill gaps from upscaling
    mask = binary_dilation(mask, circleKern(5, dims=2)[np.newaxis])

    # save data
    metadata[str(i+1).zfill(5)] = {'cell_type': cell_type,
                                   'n_thick_branches':n_thick_branches,
                                   'n_long_thin_branches':n_long_thin_branches,
                                   'n_short_thin_branches':n_short_thin_branches,
                                   'n_branches': n_thick_branches+n_long_thin_branches+n_short_thin_branches,
                                   'branches': branches}

    
    with h5py.File(f'{args.save_path}mask_{str(i+1).zfill(5)}.h5', 'w') as hf:
        hf.create_dataset('data', data=mask[np.newaxis].astype(bool))
        hf.create_dataset('instance', data=mask[np.newaxis].astype(np.uint8))

with open(f'{args.save_path}metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)





































