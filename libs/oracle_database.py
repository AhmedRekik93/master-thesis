from generators.utils import get_x_case, get_y_case
import math, random, h5py
import numpy as np

def compute_iou(y_true, y_pred):
    '''Compute the intersection over union between two masks'''
    intersection = y_true * y_pred
    intersection = intersection.sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return float(intersection) / float(union) if union != 0 else 0.

def get_start_end_liver(y):
    '''Retrieve in a volume the sub-volume interval, where the liver is featured
    @param y The CT volume
    @return the row of the starting slice
    @return the row of the ending slice
    '''
    start = 0
    end = len(y) - 1
    for i, s in enumerate(y):
        if s.max() > 0.:
            start = i
            break
    last_liver = start
    for i, s in enumerate(y[start:]):
        if s.max() > 0:
            last_liver = i
    end = last_liver + start
    if start > 5:
        start -= 2 # add some empty slices
    if end < len(y) + 4:
        end += 3
    return start, end


def extract_samples(y, cid, start, end, threshold = 0.05):
    '''Computes the training database for the oracle from one volume
    @param y The volume
    @param cid The volume id
    @param start The first row of the liver sub-volume in the CT scan
    @param start The last row of the liver sub-volume in the CT scan
    @param threshold The interleave between the IoUs of two generated samples
    '''
    scores = []
    for ki, i in enumerate(y):
        prev = None
        if i.sum() == 0:
            scores.append((cid, ki, ki, -1, 0))
            # Pick a random segmentation
            scores.append((cid, ki, random.randint(int(start), int(end)), 0, 0))
            scores.append((cid, ki, random.randint(int(start), int(end)), 0, 0))
            scores.append((cid, ki, random.randint(int(start), int(end)), 0, 0))
        else:
            for kj, j in enumerate(y[start:]):
                if kj + start > end:
                    break
                curr = compute_iou(i,j) if j.sum() > 0. else 0.
                if prev is None or math.fabs(prev - curr) >= threshold or curr == 1.:
                    prev = curr if curr != 1. else prev
                    scores.append((cid, ki, kj + start, curr, i.sum()))
    return scores

def get_oracle_training_data(cid, threshold= 0.05):
    '''Compute and retrieve the training data generated from a single volume
    @param cid The volume id
    @param threshold The interleave between the IoUs of two generated samples
    '''
    with h5py.File('data/training_data.h5', 'r') as hdf:
        x = get_x_case(hdf, cid)
        y = np.array(get_y_case(hdf, cid))
        y = np.where(y>0, 1, 0)
        print 'Volume shape: '+ str(y.shape)
        
        start, end = get_start_end_liver(y)
        print('Liver features start at slice: '+ str(start))
        print('Liver features end in slice: '+ str(end))

        print('Start sampling new samples.')
        scores = extract_samples(y, cid, start, end - 3 , threshold )
        print('%i generated training samples.'%len(scores))
        return scores