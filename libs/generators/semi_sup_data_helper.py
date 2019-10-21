import h5py, json
import numpy as np

def get_training_indices():
    with open('indices/collected_training_data_indices.json', 'r') as js:
        return json.load(js)
    
def get_collected_semi_supervised_samples_indices():
    with open('indices/collected_semi_supervised_samples.json', 'r') as js:
        return json.load(js)

def add_new_training_data(X, Y, Y_replacement = []):
    indices = get_training_indices()
    l = X.shape[0]
    prev = len(filter(lambda x: x[0] == 1, indices))
    for j in range(l):
        indices.append((1, prev + j))
    with open('indices/collected_training_data.json', 'w') as js:
        json.dump(indices, js)
    with h5py.File('data/collected_training_data.h5', 'r', libver='latest') as hdf:
        print hdf.get(str(0))
        labeled_x = np.array(hdf.get(str(0))['x'])
        labeled_y = np.array(hdf.get(str(0))['y'])
        if hdf.get(str(1)) is not None:
            collected_x = np.array(hdf.get(str(1))['x'])
            collected_y = np.array(hdf.get(str(1))['y'])
        else:
            collected_x, collected_y = None, None 
        hdf.close()
    #Takes too much space when using r+ 
    with h5py.File('data/collected_training_data.h5', 'w', libver='latest') as hdf:
        group_0 = hdf.create_group('0')
        group_0.create_dataset('x', data = labeled_x, dtype= 'float32')
        group_0.create_dataset('y', data = labeled_y, dtype = 'uint8')
        del labeled_x, labeled_y
        if len(Y_replacement) > 0:
            print 'Samples to replace: ' + str(len(Y_replacement))
            for sample in Y_replacement:
                if sample[0] < len(collected_y):
                    collected_y[sample[0]] = sample[1]
                else:
                    print 'Check this ' + str(sample[0])
                    print len(collected_y)
        size_replacement = len(Y_replacement)
        del Y_replacement
        if collected_x is None:
            collected_x = X.astype(np.float32)
            collected_y = Y.astype(np.uint8)
            old_size = 0
        else:
            old_size = len(collected_x)
            collected_x = np.concatenate((collected_x, X))
            collected_y = np.concatenate((collected_y, Y))
        del X, Y
        group_1 = hdf.create_group('1')
        group_1.create_dataset('x', data = collected_x, dtype= 'float32')
        group_1.create_dataset('y', data = collected_y, dtype = 'uint8')
        print 'Summary:'
        print 'Old size %i'%old_size
        print '%i newly added samples.'%l
        print '%i replaced samples.'%size_replacement
        print '%i new size of the collected data.'%len(collected_y)
        
        

def add_new_training_data_with_separate_nan(X, Y, X_NaN, Y_replacement = []):
    '''This function evicts all previous slices with having class'''
    indices = get_training_indices()
    l = X.shape[0]
    prev = len(filter(lambda x: x[0] == 1, indices))
    for j in range(l):
        indices.append((1, prev + j))
    indices = filter(lambda x: x[0] != 2, indices)
    
    for j in range(len(X_NaN)):
        indices.append((2, j))
        
    with h5py.File('data/collected_training_data.h5', 'r', libver='latest') as hdf:
        print hdf.get(str(0))
        labeled_x = np.array(hdf.get(str(0))['x'])
        labeled_y = np.array(hdf.get(str(0))['y'])
        if hdf.get(str(1)) is not None:
            collected_x = np.array(hdf.get(str(1))['x'])
            collected_y = np.array(hdf.get(str(1))['y'])
        else:
            collected_x, collected_y = None, None 
        hdf.close()
    #Takes too much space when using r+ 
    with h5py.File('data/collected_training_data.h5', 'w', libver='latest') as hdf:
        group_0 = hdf.create_group('0')
        group_0.create_dataset('x', data = labeled_x, dtype= 'float32')
        group_0.create_dataset('y', data = labeled_y, dtype = 'uint8')
        del labeled_x, labeled_y
        if len(Y_replacement) > 0:
            print 'Samples to replace: ' + str(len(Y_replacement))
            for sample in Y_replacement:
                if sample[0] < len(collected_y):
                    collected_y[sample[0]] = sample[1]
                else:
                    print 'Check this ' + str(sample[0])
                    print len(collected_y)
        size_replacement = len(Y_replacement)
        del Y_replacement
        if collected_x is None:
            collected_x = X.astype(np.float32)
            collected_y = Y.astype(np.uint8)
            old_size = 0
        else:
            old_size = len(collected_x)
            collected_x = np.concatenate((collected_x, X))
            collected_y = np.concatenate((collected_y, Y))
        del X, Y
        group_1 = hdf.create_group('1') # Slices with high confidence
        group_1.create_dataset('x', data = collected_x, dtype= 'float32')
        group_1.create_dataset('y', data = collected_y, dtype = 'uint8')
        group_2 = hdf.create_group('2') # Slices in the NaN class
        group_2.create_dataset('x', data = X_NaN, dtype= 'float32')
        group_2.create_dataset('y', data = np.zeros(X_NaN.shape, dtype = 'uint8'), dtype = 'uint8')
        print('Summary:')
        print('Old size %i'%old_size)
        print('%i newly added samples.'%l)
        print('%i replaced samples.'%size_replacement)
        print('%i new size of the collected data.'%len(collected_y))
    with open('indices/collected_training_data_indices.json', 'w') as js:
        print 'Save Indices'
        json.dump(indices, js)