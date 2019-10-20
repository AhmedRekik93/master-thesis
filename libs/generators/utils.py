import numpy as np
import keras
import h5py
import random

DATA_DIR = 'data/'

def get_slice(case, row,hdf = None):
    '''Get CT-scan slice with its corresponding labeling'''
    if hdf is None:
        hdf = h5py.File(DATA_DIR + 'train_data.h5', 'r')
    return get_x_slice(hdf, case, row), get_y_slice(hdf, case, row)

def get_steps(batch_size, to= 100, data='training'):
    '''Get number of steps relative to a batch size for a set of defined slices.'''
    return int(len(generate_indices(to = to, data= data)) / batch_size)

def resize_shape(arr):
    '''Resize numpy array from (X,X) to (X,X,1) for training purpose.'''
    return np.resize(arr, arr.shape + (1,))

def get_x_slice(h5py, case, row):
    '''Get slice of a given case and a row from a h5 dataset.'''
    if row < 0:
        return None
    try:
        return get_x_case(h5py, case)[row]
    except:
        print('Inexisting slice %i of patient %i!'%(row,case))
        return None

def get_x_case(h5py, case):
    '''Get volume CT-scan of a given case and a row from a h5 dataset.'''
    return h5py.get(str(case))['x']

def get_y_slice(h5py, case, row):
    '''Get slice label CT-scan of a given case and a row from a h5 dataset.'''
    if row < 0:
        return None
    try:
        return get_y_case(h5py, case)[row]
    except: 
        print('Inexisting labeling %i of patient %i!'%(row,case))
        return None
    
def get_y_case(h5py, case):
    '''Get label volume CT-scan of a given case and a row from a h5 dataset.'''
    return h5py.get(str(case))['y']

def get_case_length(h5py, case):
    '''Get number of slices present in a case.'''
    return h5py.get(str(case))['x'].shape[0]
