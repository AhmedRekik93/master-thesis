import json, h5py, keras, random
import numpy as np
from utils import get_x_slice, get_y_slice
from image_processing_utils import shift_image, rotate90, rotate_scipy
from semi_sup_data_helper import add_new_training_data_with_separate_nan

def get_training_indices():
    with open('indices/collected_training_data_indices.json', 'r') as js:
        return json.load(js)
    
def get_collected_semi_supervised_samples_indices():
    with open('indices/collected_semi_supervised_samples.json', 'r') as js:
        return json.load(js)

    
class SemiSupervisedBatchGenerator(keras.utils.Sequence):
    '''Keras data generator.
    @param data data type
    @param batch_size returned batch size
    @param shuffle Set True to shuffle dataset indices 
    '''
    def __init__(self, batch_size=8, num_classes = 1, size = 416, gaussian= False, with_augmentation = True):
        'Initialization'
        self.batch_size = batch_size
        self.noise_offset = 0.01
        self.num_classes = num_classes
        self.size = size # Default
        self.indices = get_training_indices()
        self.collected_slices = get_collected_semi_supervised_samples_indices()
        self.indices.sort()
        self.with_gaussian = gaussian
        self.noise = 0.1
        self.augment = with_augmentation
        random.shuffle(self.indices)
        self.open_files()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(len(self.indices) / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, Y = self.__data_generation(indices)
        
        return X, Y
    
    def open_files(self):
        self.hdf = h5py.File('data/collected_training_data.h5', 'r', swmr = True)
        self.extern_hdf = h5py.File('data/training_data.h5', 'r', swmr = True)
        
    def close_files(self):
        self.hdf.close()
        self.extern_hdf.close()

    def on_epoch_end(self):
        'Shuffle indices after each epoch'
        random.shuffle(self.indices)
        # Enhance noise
        if self.noise < 0.5:
            self.noise += self.noise_offset
            round(self.noise, 2) # Avoid buggy python floats
            
    def get_random_noise(self):
        '''Randomize added noise during training'''
        return random.uniform(0.1, 0.25)
        # return random.uniform(max(0, self.noise - 0.25), self.noise)
    
    def shift_slice(self, img, dx, dy):
        return shift_image(img, dx, dy)
    
    def get_sample(self, case, index):
        return get_x_slice(self.hdf, case, index)
    
    def get_label(self, case, index):
        Y = get_y_slice(self.hdf, case, index)
        Y[1<Y] = 1
        return Y
    
    def sanitize_label(self, Y):
        if self.num_classes != 1:
            Y = self.convert_ground_truth(Y)
        else:
            Y= Y.reshape((self.size, self.size, 1))            
        return Y
        
    def generate_stacked_feature_map(self, y):
        Y = np.empty((self.size, self.size, 2), dtype='float32') 
        Y[...,0] = np.where(y==2., 1., y).reshape((self.size,self.size)) # To Liver
        tmp = np.where(y==1., 0, y)
        Y[...,1] = np.where( tmp == 2., 1., tmp).reshape((self.size,self.size)) # To Tumor
        return Y
    
    def convert_ground_truth(self, y):
        try:
            y = keras.utils.to_categorical(y, num_classes=self.num_classes, dtype='float32') if self.softmax else self.generate_stacked_feature_map(y)
        except:
            y = keras.utils.to_categorical(np.where(y==3,2,y), num_classes=self.num_classes, dtype='float32') if self.softmax else self.generate_stacked_feature_map(y)
        y = y * self.weights if self.weights is not None else y
        return y
    
    def __get_unannotated_data(self, indices):
        size = self.size
        num_stacks = self.batch_size
        U = np.empty((num_stacks, size, size, 1), dtype='float32')
        for i, index  in enumerate(indices):
            U[i,] = self.get_sample(index[0], index[1])
        return U
        
    def horizontal_flip(self, sample):
        if len(sample.shape) == 2:
            return sample[:, ::-1]
        else:
            return sample[:, ::-1, :]
    
    def vertical_flip(self, sample):
        return sample[::-1]
    
    def save_slices_corpus(self):
        with open('indices/collected_semi_supervised_samples.json', 'w') as js:
            json.dump(self.collected_slices, js) 
           
        
    def search_index(self, case, sl):
        for c, value in enumerate(self.collected_slices):
            if value[0] == case and value[1] == sl:
                return c
        print 'What the Heck, this should not happen!!!'
        return -1 
        
    def count_new_samples(self, new_slices):
        X = []
        Y = []
        X_NaN = []
        Y_replacement = []
        existing = {}
        for sample in filter(lambda x: x[0] != 0, self.collected_slices): # Sample = (case, slice, score)
            if sample[0] in existing:
                existing[sample[0]][sample[1]] = sample[2]
            else:
                existing[sample[0]] = { sample[1]: sample[2] }
        counter = 0
        for i, it in enumerate(new_slices):
            if it[0] in existing and it[1] in existing[it[0]]:
                counter += 1
        return len(new_slices) - counter
        
    def add_separate_samples(self, new_slices, nan_slices):
        '''Create new database with the semi-supervised data.
        @param new_slices The samples (axial slice, computed mask) with high confidence
        @param nan_slices The samples with an undefined IoU
        '''
        # Add samples having JC above 0.9 separately from slices that have NaN JS
        # new slices have the format [(case, slice, pseudo-label, score)]
        X = []
        Y = []
        X_NaN = []
        Y_replacement = []
        existing = {}
        for sample in filter(lambda x: x[0] != 0, self.collected_slices): # Sample = (case, slice, score)
            if sample[0] in existing:
                existing[sample[0]][sample[1]] = sample[2]
            else:
                existing[sample[0]] = { sample[1]: sample[2] }
        
        # When using a vanishing database or bucket classification, ignore this loop
        if False: 
            for i, it in enumerate(new_slices):
                if it[0] in existing and it[1] in existing[it[0]]:
                    # Existing labels
                    print 'Overlapping segmentation found for Slice (%i, %i)!'%(it[0], it[1])
                    existing[it[0]][it[1]] = it[3]
                    t = self.search_index(it[0], it[1])
                    self.collected_slices[t] = [it[0], it[1], it[3]]
                    Y_replacement.append((t, it[2]))
                    continue

                self.collected_slices.append([it[0], it[1], it[3]]) # [case, slice, score]
                X.append(get_x_slice(self.extern_hdf, it[0], it[1]))
                Y.append(it[2]) 
            
        for i, it in enumerate(nan_slices):
            X_NaN.append(get_x_slice(self.extern_hdf, it[0], it[1]))
        
        X = np.array(X, dtype= 'float32')
        Y = np.array(Y, dtype= 'uint8')
        X_NaN = np.array(X_NaN, dtype= 'float32')
        
        self.close_files()
        if X.shape[0] > 0 or len(Y_replacement) > 0:
            add_new_training_data_with_separate_nan(X, Y, X_NaN, Y_replacement)
            self.save_slices_corpus()
            self.indices = get_training_indices()
        random.shuffle(self.indices)
        self.open_files()
        
    def __data_generation(self, indices):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        size = self.size
        num_stacks = self.batch_size
        X = np.empty((num_stacks, size, size, 1), dtype='float32')
        Y = np.empty((num_stacks, size, size, self.num_classes), dtype='float32')
        C = np.empty((num_stacks, 1), dtype='float32')
        
        for i, index  in enumerate(indices):
            # Sample getter
            x = self.get_sample(index[0], index[1])
            y = self.get_label(index[0], index[1])
            if self.augment: 
                if random.randint(0, 2) == 0:
                    # 1/3 prob to perform horizontal flipping
                    x = self.horizontal_flip(x)
                    y = self.horizontal_flip(y)
                if random.randint(0, 2) == 0:
                    x = self.vertical_flip(x)
                    y = self.vertical_flip(y)
                    
                dx, dy = random.randint(-5, 5), random.randint(-5, 5)
                x = self.shift_slice(x, dx, dy)
                y = self.shift_slice(y, dx, dy)
                r = random.randint(0,3)
                x = rotate90(x, r)
                y = rotate90(y, r)
                theta = random.randint(-10, 10)
                x = rotate_scipy(x, theta)
                y = rotate_scipy(y, theta)
            X[i,] = add_gaussian_noise(x, self.get_random_noise()) if (self.with_gaussian and random.randint(0,1) == 1) else x
            Y[i,] = self.sanitize_label(y)
        return X, Y