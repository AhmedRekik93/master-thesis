import numpy as np

def depth_checking(stack, cl =2, label = 1):
    '''Apply depth checking on a sub-volume of 5 axial slices. The returned inference corresponds to the axial slice in the middle (pos 2)
    @param stack The sub-volume
    @param cl The position of the middle axial slice
    @param label The type of the prediction 1 = liver, 2 = tumor
    @return post-processed segmentation
    '''
    s = np.sum(stack[...,cl], axis=0)
    pp = np.where(label * 3<=s, 1, 0)
    back_forth = stack[1,..., cl] + stack[3,..., cl]
    return np.where(back_forth == 2, 1, pp)

def self_ensembling(sample, model):
    '''Ensemble the inferences of an image under four rotations
    @param sample The unsupervised data
    @param model A model that infer segments
    @return self-ensembled inference 
    '''
    buf = []
    for i in range(4):
        y = model.predict(np.rot90(sample, i).reshape(1, 416, 416, 1)).reshape(416, 416)
        buf.append(np.rot90(y, 4 - i))
    tot = np.add(np.add(buf[0], buf[1]), np.add(buf[2], buf[3]))
    tot = tot / 4
    return np.round(tot)