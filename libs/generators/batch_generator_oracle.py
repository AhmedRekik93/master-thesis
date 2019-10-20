from image_processing_utils import shift_image, rotate90
from utils import get_x_slice, get_y_slice
import json, keras, math, random, h5py
import numpy as np

def get_samples_indices():
    with open('indices/training_oracle_model.json', 'r') as js:
        return json.load(js)

class BatchGeneratorTrainOracle(keras.utils.Sequence):
    '''Keras data generator.
    @param data data type
    @param batch_size returned batch size
    @param shuffle Set True to shuffle dataset indices 
    '''
    # Modality should be classification, regression or buckets
    def __init__(self, batch_size=8, num_classes = 1, size = 416, gaussian= False, with_augmentation = False, modality = 'buckets', vgg = False):
        'Initialization'
        self.batch_size = batch_size
        self.noise_offset = 0.01
        self.num_classes = num_classes
        self.size = size # Default
        indices = get_samples_indices()
        s = int(len(indices) * 0.85)
        self.indices = indices[:s]
        self.with_gaussian = gaussian
        self.noise = 0.1
        self.augment = with_augmentation
        random.shuffle(self.indices)
        self.open_files()
        self.modality = modality
        self.vgg = vgg

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
        self.hdf = h5py.File('data/training_data.h5', 'r', swmr = True)
        
    def close_files(self):
        self.hdf.close()

    def on_epoch_end(self):
        'Shuffle indices after each epoch'
        random.shuffle(self.indices)
        # Enhance noise
        if self.noise < 0.5:
            self.noise += self.noise_offset
            round(self.noise, 2) # Avoid buggy python floats
            
    def get_random_noise(self):
        '''Randomize added noise during training'''
        return random.uniform(max(0, self.noise - 0.25), self.noise)
    
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
    
    def jaccard_to_class(self, score):
        if score == -1:
            return 0 # NaN
        elif 0 <= score and score < 0.4:
            return 1
        elif 0.4 <= score and score < 0.6:
            return 2
        elif 0.6 <= score and score < 0.75:
            return 3
        elif 0.75 <= score and score < 0.9:
            return 4
        elif 0.9 <= score and score <= 1.:
            return 5
        else:
            print 'ERROR!'
            print 'Score: ' + str(score)
            -132141414531 # Should raise an Error
            
    def get_output_size(self):
        if self.modality == 'regression':
            return 2
        elif self.modality == 'classification':
            return 102
        else:
            return 6
        
    def __data_generation(self, indices):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        size = self.size
        num_stacks = self.batch_size
        c = 3 if self.vgg else 2
        X = np.empty((num_stacks, size, size, c), dtype='float32')
        Y = np.empty((num_stacks, self.get_output_size()), dtype='float32') 
        
        for i, index  in enumerate(indices):
            # Sample getter
            x = self.get_sample(index[0], index[1]) # The Observation
            y = self.get_label(index[0], index[2]) # Its segmentation
            if self.augment: 
                if random.randint(0, 2) == 0:
                    # 1/4 prob to perform horizontal flipping
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
                
            y = self.sanitize_label(y)
            y = y * 1.
            y = x + y
            if not self.vgg:
                X[i,] = np.concatenate((x, y.astype('float32')), axis = -1)
            else:
                filler = np.full(x.shape, x.mean(), 'float32')
                X[i,] = np.concatenate((x, y.astype('float32'), filler), axis = -1)
            if self.modality == 'buckets':
                gt = np.zeros(6).astype('float32')
                gt[self.jaccard_to_class(index[3])] = 1.
                Y[i] = gt
            elif self.modality == 'regression':
                # set NaN slices as 1
                if index[3] != -1:
                    Y[i] = np.array([1., index[3]]).astype('float32')
                else: 
                    Y[i] = np.array([0., 0.]).astype('float32')
            else:           
            # TO 102 Classes implementation
                gt = np.zeros(102).astype('float32')
                k = index[3]
                if k == -1:
                    gt[101] = 1.
                else:
                    k = int(round(k * 100))
                    gt[k] = 1.
                Y[i] = gt
        X = (X - X.mean()) / X.std()
        return X, Y
    
class BatchGeneratorValidationOracle(keras.utils.Sequence):
    '''Keras data generator.
    @param data data type
    @param batch_size returned batch size
    @param shuffle Set True to shuffle dataset indices 
    '''
    def __init__(self, batch_size=8, num_classes = 1, size = 416, modality = 'buckets', vgg= False):
        'Initialization'
        self.batch_size = batch_size
        self.noise_offset = 0.01
        self.num_classes = num_classes
        self.size = size # Default
        self.indices = get_samples_indices()
        indices = get_samples_indices()
        s = int(len(indices) * 0.85)
        self.indices = indices[s:]
        random.shuffle(self.indices)
        self.open_files()
        self.modality = modality
        self.vgg = vgg


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
        self.hdf = h5py.File('data/training_data.h5', 'r', swmr = True)
        
    def close_files(self):
        self.hdf.close()

    def on_epoch_end(self):
        'Shuffle indices after each epoch'
        random.shuffle(self.indices)
            
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
    
    def convert_ground_truth(self, y):
        try:
            y = keras.utils.to_categorical(y, num_classes=self.num_classes, dtype='float32') if self.softmax else self.generate_stacked_feature_map(y)
        except:
            y = keras.utils.to_categorical(np.where(y==3,2,y), num_classes=self.num_classes, dtype='float32') if self.softmax else self.generate_stacked_feature_map(y)
        y = y * self.weights if self.weights is not None else y
        return y
    
    def jaccard_to_class(self, score):
        if score == -1:
            return 0 # NaN
        elif 0 <= score and score < 0.4:
            return 1
        elif 0.4 <= score and score < 0.6:
            return 2
        elif 0.6 <= score and score < 0.75:
            return 3
        elif 0.75 <= score and score < 0.9:
            return 4
        elif 0.9 <= score and score <= 1.:
            return 5
        else:
            print 'What the F***!'
            print 'Score: ' + str(score)
            -132141414531 # Should raise an Error
            
    def get_output_size(self):
        if self.modality == 'regression':
            return 2
        elif self.modality == 'classification':
            return 102
        else:
            return 6
        
    def __data_generation(self, indices):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        size = self.size
        num_stacks = self.batch_size
        c = 3 if self.vgg else 2
        X = np.empty((num_stacks, size, size, c), dtype='float32')
        Y = np.empty((num_stacks, self.get_output_size()), dtype='float32') 
        
        for i, index  in enumerate(indices):
            # Sample getter
            x = self.get_sample(index[0], index[1]) # The Observation
            y = self.get_label(index[0], index[2]) # Its segmentation
            
            y = self.sanitize_label(y)
            y = y * 1.
            y = x + y
            if not self.vgg:
                X[i,] = np.concatenate((x, y.astype('float32')), axis = -1)
            else:
                filler = np.full(x.shape, x.mean(), 'float32')
                X[i,] = np.concatenate((x, y.astype('float32'), filler), axis = -1)            
            if self.modality == 'buckets':
                gt = np.zeros(6).astype('float32')
                gt[self.jaccard_to_class(index[3])] = 1.
                Y[i] = gt
            elif self.modality == 'regression':
                if index[3] != -1:
                    Y[i] = np.array([1., index[3]]).astype('float32')
                else: 
                    Y[i] = np.array([0., 0.]).astype('float32')
            else:           
            # TO 102 Classes implementation
                gt = np.zeros(102).astype('float32')
                k = index[3]
                if k == -1:
                    gt[101] = 1.
                else:
                    k = int(round(k * 100))
                    gt[k] = 1.
                Y[i] = gt
        return X, Y
    