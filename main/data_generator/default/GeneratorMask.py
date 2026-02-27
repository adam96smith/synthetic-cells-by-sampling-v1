'''
Image Label Generator

NOTE: This is to be customised between datasets. The goal is to match the size, numbers, 
      density and shape of the cells in the real images. Branches and Invaginations might
      be key features to add.

Output: .h5 file of binary image ('data') and labelled image ('instance') required for 
        GeneratorImage.py and GeneratorCurveSeg.py (if applicable).

'''

import numpy as np
import os
import h5py
from tqdm import tqdm
import pickle
import sys

if os.getcwd() not in sys.path:
    (sys.path).append(os.getcwd())

# import some functions from my_elektronn3
import my_elektronn3
from my_elektronn3.custom.perlin_noise import *
from my_elektronn3.custom.curvature import *
from my_elektronn3.data import transforms

from scipy.ndimage import distance_transform_edt
from skimage.measure import label
from utils import load_config, nearest_power_of_two

import argparse

parser = argparse.ArgumentParser(description='Generate Synthetic Cell Shapes.')
parser.add_argument('--save-path', type=str, default='synthetic_train/', 
                    help='Output Directory')
parser.add_argument('--N', type=int, default=100, 
                    help='Number of Total Generated Samples.')
parser.add_argument('--dataset-id', type=str, required=True,
                    help='Short identifier for the dataset (e.g., H157). Used for logging, config selection, etc.')
parser.add_argument('--config', type=str, required=True,
                    help='Config. File for Synthetic Images')
parser.add_argument('--global-config', type=str, default=None,
                    help='Config. File for Sampler')
args = parser.parse_args()

''' PARAMETERS '''

config = load_config(args.config)
if args.global_config is None:
    global_params = load_config(f'config/{args.dataset_id}/global_parameters.yaml')
else:
    global_params = load_config(args.global_config)

# Parameters
sampling = global_params['SAMPLING']
aniso_factor = nearest_power_of_two(sampling[0]/sampling[1])

zres, xres, yres  = config['SHAPE']['IMAGE_SIZE']
sf  = config['SHAPE']['DOWNSAMPLE']
n0, n1 = config['SHAPE']['CELL_N']
r0, r1 = config['SHAPE']['CELL_R']
z_scale = config['SHAPE']['Z_SCALE']
r_sep = config['SHAPE']['CELL_SEPARATION']
P = config['SHAPE']['CELL_COUPLING_CHANCE']
ept1 = config['SHAPE']['DEFORM_1']
ept2 = config['SHAPE']['DEFORM_2']

if sf > 1:
    sampling = (sampling[0], sf*sampling[1], sf*sampling[2])

## Define Box in Image to Add Cells such that none cross the boarder (essential for sampling method)
z0 = int(z_scale*r1/sampling[0]); z1 = zres - z0
x0 = int(r1/sampling[1]); x1 = xres - x0
y0 = int(r1/sampling[2]); y1 = yres - y0

# ## pad based on max. deformation
# padz = 0; padx = 0; pady = 0;
# if ept1 is not False:
#     padz = max(padz, int(ept1[1]*zres)); padx = max(padx, int(ept1[1]*xres)); pady = max(pady, int(ept1[1]*yres))
# if ept2 is not False:
#     padz = max(padz, int(ept2[1]*zres)); padx = max(padx, int(ept2[1]*xres)); pady = max(pady, int(ept2[1]*yres))
# z0 += padz; z1 += padz; zres += 2*padz
# x0 += padx; x1 += padx; xres += 2*padx
# y0 += pady; y1 += pady; yres += 2*pady

print(f'Z: {z0} - {z1}')
print(f'X: {x0} - {x1}')
print(f'Y: {y0} - {y1}')

''' MAIN '''

assert args.save_path[-1] == '/'
os.makedirs(args.save_path, exist_ok=True)

if ept1 is not False:
    EPT1 = transforms.ElasticPerlinTransform(prob=1,
                                             grid_points=ept1[0],
                                             p=ept1[1],  order=1) # small
if ept2 is not False:
    EPT2 = transforms.ElasticPerlinTransform(prob=1,
                                             grid_points=ept2[0],
                                             p=ept2[1],  order=1) # large

# initialise image and sample space
Z, X, Y = np.meshgrid(np.arange(0,zres), np.arange(0,xres), np.arange(0,yres), indexing='ij')
sZ, sX, sY = np.meshgrid(np.arange(z0,z1), np.arange(x0,x1), np.arange(y0,y1), indexing='ij')

metadata = {}
for i in tqdm(range(args.N), desc='Generating Image Masks'):

    N = np.random.choice(np.arange(n0,n1+1))

    # radius of first cell
    r_c = np.random.uniform(r0,r1)

    # initiate sample space
    sample_space = np.ones(sZ.shape, bool)
    
    centroids = []
    radii = []
    counter = 0
    while counter < N and sample_space.sum() > 0:

        radii.append(r_c) # add cell to metadata
    
        # all possible coordinates for new cell
        z_space = sZ[sample_space];  x_space = sX[sample_space]; y_space = sY[sample_space]
        coords = np.vstack((z_space,x_space,y_space))
        
        # randomly select a coordinate
        j = np.random.choice(coords.shape[1])         
        z_c = z_space[j];  x_c = x_space[j]; y_c = y_space[j]
    
        centroids.append([z_c, x_c, y_c])
    
        if counter == 0:
            dist = np.sqrt(((Z-z_c)*sampling[0]/z_scale)**2+((X-x_c)*sampling[1])**2+((Y-y_c)*sampling[1])**2) - r_c
            dist_s = np.sqrt(((sZ-z_c)*sampling[0]/z_scale)**2+((sX-x_c)*sampling[1])**2+((sY-y_c)*sampling[1])**2) - r_c
        else:
            dist = np.minimum(dist, np.sqrt(((Z-z_c)*sampling[0]/z_scale)**2+((X-x_c)*sampling[1])**2+((Y-y_c)*sampling[1])**2) - r_c)
            dist_s = np.minimum(dist_s, np.sqrt(((sZ-z_c)*sampling[0]/z_scale)**2+((sX-x_c)*sampling[1])**2+((sY-y_c)*sampling[1])**2) - r_c)
    
        counter += 1
    
        # next cell to add        
        r_c = np.random.uniform(r0,r1)

        if np.random.rand() < P: # couple cells
            sample_space = np.abs(dist_s - .8*r_c) <= sampling[0] 
            # sample_space = np.abs(dist_s - r_sep) <= sampling[0] 
        else:
            sample_space = dist_s > r_sep


    # Generate Mask labels - split membrane and reassign nearest interior label
    outline = (-3<dist)&(dist<=0) # not sure if 3 is a robust choice
    interior = label(dist<=-3)
    
    dist_map, indices = distance_transform_edt(interior==0, sampling=sampling, return_indices=True)
    
    mask_labels = (dist<=0) * interior[tuple(indices)]

    # deform labelled image
    if ept1 is not False:
        mask_labels = EPT1(mask_labels[np.newaxis], [])[0][0] # small
    if ept2 is not False:
        mask_labels = EPT2(mask_labels[np.newaxis], [])[0][0] # large

    # # remove padding
    # if padz > 0:
    #     mask_labels = mask_labels[aniso_factor*padz:-aniso_factor*padz,:,:]
    # if padx > 0:
    #     mask_labels = mask_labels[:,padx:-padx,:]
    # if pady > 0:
    #     mask_labels = mask_labels[:,:,pady:-pady]
        
    
    with h5py.File(f'{args.save_path}mask_{str(i+1).zfill(5)}.h5', 'w') as hf:
        dataset = hf.create_dataset('data', data=(mask_labels[np.newaxis]>0.5).astype(bool))

        dataset = hf.create_dataset('instance', data=mask_labels[np.newaxis].astype(np.uint8))

    metadata[str(i+1).zfill(5)] = {'N': N,
                                   'radii': radii, 
                                   }

with open(f'{args.save_path}metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

















## OLD STUFF

# ''' FUNCTIONS '''

# def generate_centroids(N, S, distance, P=1, tol=2.0):
#     '''
#     Centroids are generated based using the following algorithm:

#         1) First centroid → placed randomly.
    
#         2) Next centroid:
        
#             With probability P,  placed approx. distance from one existing cell (forming a cluster)
            
#             Otherwise, placed randomly (but still outside 2R of any existing cell).
        
#         4) No centroid may be placed within 2R of any existing one.
    
#     For each centroid, a sample space is calculated. If the space is empty, we return the 
#     current state.
    
#     '''

#     z0, x0, y0, z1, x1, y1 = S
    
#     Z,X,Y = np.meshgrid(np.arange(z0,z1),np.arange(x0,x1),np.arange(y0,y1),indexing='ij')
#     centroids = []
#     n = 0

#     def get_S():
#         if n == 0:
#             return np.ones([z1-z0,x1-x0,y1-y0], bool)
#         else:
#             # calculate radial distance to closest centroid
#             nearest_centroid = 1000*np.ones([z1-z0,x1-x0,y1-y0])
#             for zc,xc,yc in centroids:
#                 nearest_centroid = np.minimum(nearest_centroid, 
#                                               np.sqrt((Z-zc)**2+(X-xc)**2+(Y-yc)**2))
            
#             if np.random.rand() < P: # 
#                 return ((distance-tol)<nearest_centroid)*(nearest_centroid<(distance+tol))
#             else:
#                 return ((distance+tol)<nearest_centroid)
                
    
#     while n < N:

#         sample_space = get_S()

#         if sample_space.sum() > 0: # we can sample a new cell
            
#             # all possible coordinates for new cell
#             z_space = Z[sample_space];  x_space = X[sample_space]; y_space = Y[sample_space]
#             coords = np.vstack((z_space,x_space,y_space))
            
#             # randomly select a coordinate
#             i = np.random.choice(coords.shape[1])         
#             z = z_space[i];  x = x_space[i]; y = y_space[i]
            
#             # save new centroid
#             centroids.append(np.array([z, x, y]))
#             n += 1
#         else:
#             print('Terminating Prematurely')
#             n += 10000 ## terminate simulation

#     return np.array(centroids)

# def add_cells_to_volume(shape, centroids, radii, z_scale=1, voxel_spacing=(1,1,1)):    
    
#     Z, X, Y = np.meshgrid(np.arange(shape[0]), 
#                           np.arange(shape[1]),
#                           np.arange(shape[2]), 
#                           indexing='ij')

#     volume_list = []
#     for i, (c,r) in enumerate(zip(centroids,radii)):

#         cz,cx,cy = c # centroid coordinates

#         V = np.sqrt(((Z-cz)/z_scale)**2+(X-cx)**2+(Y-cy)**2) < r

#         volume_list.append(((i+1)*V).astype(int))

#     labelled_volume = np.array(volume_list).max(axis=0)

#     distances = np.zeros(labelled_volume.shape, np.float32)
#     for vol in volume_list:
#         distances = np.maximum( distance_transform_edt(vol!=0, sampling=voxel_spacing), distances)
    
#     markers = np.zeros_like(labelled_volume)
    
#     for i, (z, x, y) in enumerate(centroids):
#         markers[z,x,y] = i+1
    
#     output = watershed(-distances, markers, mask = labelled_volume!=0)
        
#     return output






















