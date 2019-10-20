import numpy as np
from scipy.ndimage.interpolation import zoom
from scipy.ndimage import rotate

def sagittal_to_transverse(vol):
    '''LiTS images are based 512 sagittal cut slices with X width (512, 512, X), we flip slices to (X, 512, 512) to obtain X transverse cuts'''
    return np.moveaxis(vol, -1, 0)
    
def add_gaussian_noise(inp, expected_noise_ratio=0.2):
        image = inp.copy()
        if len(image.shape) == 2:
            row,col= image.shape
            ch = 1
        else:
            row,col,ch= image.shape
        mean = 0.
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col)) * expected_noise_ratio
        gauss = gauss.reshape(row,col, 1)
        noisy = image + gauss
        return noisy
    
def rotate90(inp, direction= 0):
    '''
    Directions counter clock wise: 
    0 = Identity
    1 = pi/2 
    2 = pi
    3 = 3p/2
    
    '''
    if direction<= 0 or direction > 3:
        return inp
    else:
        return np.rot90(inp, k= direction)
    
    
def rotate_scipy(inp, theta):
    return rotate(inp, theta, reshape = False, cval= np.min(inp))

def shift_image(X, dx=32, dy= 32):
    m = np.min(X)
    X = np.roll(np.roll(X, dx, axis=1), dy, axis=0)
    if dy>0:
        X[:dy, :] = m
    elif dy<0:
        X[dy:, :] = m
    if dx>0:
        X[:, :dx] = m
    elif dx<0:
        X[:, dx:] = m
    return X

def neighborhood_labeling(stack, cl =2, label = 1):
    '''Suppose that the stack of size 5'''
    s = np.sum(stack[...,cl], axis=0)
    pp = np.where(label * 3<=s, 1, 0)
    back_forth = stack[1,..., cl] + stack[3,..., cl]
    return np.where(back_forth == 2, 1, pp)