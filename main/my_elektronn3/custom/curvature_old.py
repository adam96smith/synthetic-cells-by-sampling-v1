'''

Curvature Module (January 2025)

Contains:
    Curvature Function for Binary Mask
    Mean Negative Curvature of Mask
    Circle Kernel (sample as skimage.morphology.ball ?)
    Curvature Segmentation using NumPy and PyTorch

'''

import numpy as np

import time

# morphology 
from scipy.ndimage import convolve, binary_dilation, binary_erosion, zoom
from skimage.morphology import remove_small_objects

# pytorch for optimised functions
import torch
import torch.nn.functional as F

## Misc

def gradH(mask, aniso_factor=1.0, axis=0):
    """
    Applies separable filters to a 3D array (mask) to compute a gradient-like operation.

    Parameters:
        mask (ndarray): Input 3D array.
        axis (int): The axis along which the gradient filter is applied (0, 1, or 2).

    Returns:
        ndarray: Result of applying the gradient-like operation.
    """
    if mask.ndim != 3:
        raise ValueError("This function supports 3D arrays only.")
    if axis not in [0, 1, 2]:
        raise ValueError("Invalid axis; must be 0, 1, or 2.")

    # Define the filters
    filter_1a = np.array([1, 4, 6, 4, 1]) / 16
    filter_1b = np.array([1, 4, 6, 4, 1]) / 16
    filter_2 = np.array([-1, 0, 0, 0, 0, 0, 1]) / 6
    
    if axis==0:
        filter_2 /= aniso_factor

    result = mask.copy()

    # Apply filters
    for filt, i in zip([filter_1a, filter_1b, filter_2], [ax for ax in range(mask.ndim) if ax != axis] + [axis]):
        # Reshape filter to match the current axis
        if i == 0:
            filt = filt.reshape((len(filt), 1, 1))
        elif i == 1:
            filt = filt.reshape((1, len(filt), 1))
        elif i == 2:
            filt = filt.reshape((1, 1, len(filt))) 

        # Apply the filter using convolution
        result = convolve(result, filt, mode='nearest')

    return result
    

def curvature_of_binary_mask(arr, sampling=(1.0,1.0,1.0)):

    grad_z = gradH(arr.astype(np.float32), axis=0)
    grad_x = gradH(arr.astype(np.float32), axis=1)
    grad_y = gradH(arr.astype(np.float32), axis=2)
    
    grad_zz = gradH(grad_z, aniso_factor=sampling[0], axis=0)
    grad_xx = gradH(grad_x, aniso_factor=sampling[1], axis=1)
    grad_yy = gradH(grad_y, aniso_factor=sampling[2], axis=2)
    
    output = -(grad_zz + grad_xx + grad_yy)
    output[arr == 1] = 0
    
    return output

def mean_negative_curvature(arr, sampling=(1.0,1.0,1.0), padding=5):

    arr = np.pad(arr, ((padding,padding), (padding,padding), (padding,padding)))
    arr_dilated = binary_dilation(arr)
    
    curv = -curvature_of_binary_mask(arr, sampling = (z_sample, x_sample, y_sample))
    curv *= arr_dilated

    return np.mean(np.abs(curv)[curv<0])

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

## Curvature Segmentation with NumPy

def curvature_segmentation(mask, threshold=0.02, kernel_size = 5, iters = 3, aniso_factor=1):

    '''
    Segments mask into high-positive curved regions (and otherwise).

    The segmentation region is defined by method:
        1. Calculate the curvature of the mask using the method from Lutton et al. 2021
        2. Threshold segmentation to isolate high-positive curved regions
        3. Dilate the regions to provide a more significant segmentation
    
    Parameters:
        threshold: Threshold value for the initial segmentation (make sure this is set 
                   for different data)
        kernel_size: size of the circle/spherical kernel to use during dilation
        xy_dilate, z_dilate: how many iterations to apply dilation. xy does planar dilation 
                             whereas z does 3d dilation (!!)
        aniso_factor: Anisotropic factor of the data

    NOTE: Dilation of the segmented region should follow approximately with the anisotropy. 
          For example, if the aniso_factor is 2, then the dilation in the x-y direction 
          should be approximately twice the size than the dilation in the z-direction. It
          is not essential, but consider that when producing the segmentation with large 
          aniso_factor.

    '''

    curvature = curvature_of_binary_mask(mask, sampling=(aniso_factor, 1, 1))

    curv_mask = 1*curvature > threshold
    curv_mask[curv_mask<0] = 0
    
    curv_mask = binary_dilation(curv_mask,circleKern(kernel_size),iterations=3)
    curv_mask = binary_erosion(curv_mask,circleKern(kernel_size),iterations=3)
    
    curv_mask = remove_small_objects(curv_mask, 10)
    
    curv_mask = binary_dilation(curv_mask,circleKern(kernel_size),iterations=iters)

    return curv_mask, ~curv_mask


## Curvature Segmentation with PyTorch

def morphological_operation(mask, kernel, operation="dilation", padding_value=0, iters=1):
    """
    Perform morphological operations (dilation/erosion) with custom padding.
    
    Args:
        mask (torch.Tensor): Input binary mask (shape: [D, H, W]), should be boolean or float.
        kernel (torch.Tensor): Kernel for morphological operation, should be float.
        operation (str): Either "dilation" or "erosion".
        padding_value (int): Value for padding (0 or 1).
        iters (int): Number of iterations.
        
    Returns:
        torch.Tensor: Processed mask after dilation/erosion.
    """
    # Ensure mask and kernel are of the same floating-point type
    mask = mask.float(); kernel = kernel.float()

    if operation == "erosion":
        mask = 1-mask
    
    # Move kernel to the same device as the mask
    kernel = kernel.to(mask.device)
    
    pad_size = kernel.shape[-1] // 2

    # Add batch and channel dimensions
    mask = mask.unsqueeze(0).unsqueeze(0); kernel = kernel.unsqueeze(0).unsqueeze(0)

    for _ in range(iters):
        # Apply custom padding
        mask = F.pad(mask, (pad_size, pad_size, pad_size, pad_size, pad_size, pad_size), value=padding_value)
        
        # Perform operation
        if operation == "dilation":
            mask = F.conv3d(mask, kernel, padding=0) > 0  # Result is boolean
        elif operation == "erosion":
            mask = F.conv3d(mask, kernel, padding=0) > 0

        mask = mask.float()

    if operation == "erosion":
        mask = 1-mask
    return mask.squeeze(0).squeeze(0).bool()
    

def curvature_segmentation_torch(mask, threshold=0.02, kernel_size = 5, iters = 3, aniso_factor=1):

    '''
    Segments mask into high-positive curved regions (and otherwise).

    The segmentation region is defined by method:
        1. Calculate the curvature of the mask using the method from Lutton et al. 2021
        2. Threshold segmentation to isolate high-positive curved regions
        3. Dilate the regions to provide a more significant segmentation
    
    Parameters:
        threshold: Threshold value for the initial segmentation (make sure this is set 
                   for different data)
        kernel_size: size of the circle/spherical kernel to use during dilation
        xy_dilate, z_dilate: how many iterations to apply dilation. xy does planar dilation 
                             whereas z does 3d dilation (!!)
        aniso_factor: Anisotropic factor of the data

    NOTE: Dilation of the segmented region should follow approximately with the anisotropy. 
          For example, if the aniso_factor is 2, then the dilation in the x-y direction 
          should be approximately twice the size than the dilation in the z-direction. It
          is not essential, but consider that when producing the segmentation with large 
          aniso_factor.

    '''
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    curvature = curvature_of_binary_mask(mask.cpu().numpy(), sampling=(aniso_factor, 1, 1))

    curv_mask = 1*curvature > threshold
    curv_mask[curv_mask<0] = 0

    # Convert mask to torch tensor
    curv_mask = torch.tensor(curv_mask, dtype=torch.bool, device=device)
    kernel = torch.tensor(circleKern(kernel_size), dtype=torch.float32, device=device)

    curv_mask = morphological_operation(curv_mask, kernel, operation="dilation", iters=3)
    curv_mask = morphological_operation(curv_mask, kernel, operation="erosion", iters=3)
    
    curv_mask = remove_small_objects(curv_mask.cpu().numpy(), 10)

    curv_mask = torch.tensor(curv_mask, dtype=torch.bool, device=device)    
    curv_mask = morphological_operation(curv_mask, kernel, operation="dilation", iters=iters)

    curv_mask = curv_mask.squeeze(0).squeeze(0)

    return curv_mask.bool(), ~curv_mask.bool()


### Custom Codes for Filogen Segmentation
# NumPy

def filogen_mask_curvature_segmentation(mask, verbose=False):
    '''
    Interpolation, Thresholding and Post-Processing for Filogen Masks
    '''

    start_time = time.time()
    
    zoomed_mask = zoom(mask, (10,1,1), order=1)
    zoomed_mask = binary_dilation(zoomed_mask, circleKern(3))
    zoomed_mask = binary_erosion(zoomed_mask, circleKern(3))
    
    end_time = time.time()
    if verbose:
        print(f'Scaling Time: {end_time-start_time} seconds')

    start_time = time.time()
    
    curv_mask, _ = curvature_segmentation(zoomed_mask[::2,::2,::2], threshold=0.02, kernel_size = 5, iters = 3, aniso_factor=1)
    
    curv_mask = zoom(curv_mask, (2,2,2), order=1)
    curv_mask = binary_dilation(curv_mask, circleKern(3))
    curv_mask = binary_erosion(curv_mask, circleKern(3))
    
    end_time = time.time()
    if verbose:
        print(f'Segmentation Time: {end_time-start_time} seconds')

    return curv_mask[::10]

# PyTorch

def filogen_mask_curvature_segmentation_torch(mask, verbose=False):
    '''
    Interpolation, Thresholding and Post-Processing for Filogen Masks
    '''
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    start_time = time.time()
    
    zoomed_mask = zoom(mask, (10,1,1), order=1)

    # Convert mask to torch tensor
    zoomed_mask = torch.tensor(zoomed_mask, dtype=torch.bool, device=device)
    kernel = torch.tensor(circleKern(3), dtype=torch.float32, device=device)

    zoomed_mask = morphological_operation(zoomed_mask, kernel, operation="dilation")
    zoomed_mask = morphological_operation(zoomed_mask, kernel, operation="erosion")

    
    end_time = time.time()
    if verbose:
        print(f'Scaling Time: {end_time-start_time} seconds')

    start_time = time.time()
    
    curv_mask, _ = curvature_segmentation_torch(zoomed_mask[::2,::2,::2], threshold=0.02, kernel_size = 5, iters = 3, aniso_factor=1)

    # Convert mask to torch tensor
    curv_mask = zoom(curv_mask.cpu().numpy(), (2,2,2), order=1)

    curv_mask = torch.tensor(curv_mask, dtype=torch.bool, device=device)

    curv_mask = morphological_operation(curv_mask, kernel, operation="dilation")
    curv_mask = morphological_operation(curv_mask, kernel, operation="erosion")
    
    
    end_time = time.time()
    if verbose:
        print(f'Segmentation Time: {end_time-start_time} seconds')

    return curv_mask[::10].cpu().numpy()
    