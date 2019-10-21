import h5py, json
from libs.generators.utils import get_x_case, get_y_case

def init_ssl_databse(cid = 0):
    indices = [] 
    with h5py.File('data/training_data.h5', 'r') as hdf:
        cid = cid # 
        X = get_x_case(hdf, cid)
        Y = get_y_case(hdf, cid)
        for i in range(len(X)):
            indices.append((0, i))
        with h5py.File('data/collected_training_data.h5', 'w') as hdf_ssl:
            groupe = hdf_ssl.create_group(str(0))
            groupe.create_dataset('x', data = X, dtype='float32')
            groupe.create_dataset('y', data = Y, dtype='uint8')
        with open('indices/collected_training_data_indices.json', 'w') as js:
            json.dump(indices, js) 
        with open('indices/collected_semi_supervised_samples.json', 'w') as js:
            json.dump([], js)