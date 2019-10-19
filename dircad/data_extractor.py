import os, zipfile, numpy
from scipy.misc import imsave


patient_dir = 'PATIENT_DICOM'
masks_dir = 'MASKS_DICOM'

debug = False

def extract_zip(filename_path, dst):
    ''' Dezip a compressed zip file
    @param filename_path file to dezip (eventually its path)
    @param dst destination of the decompressed data
    '''
    p = filename_path
    if p.endswith('.zip')== False:
        p = p + '.zip'
    zip_ref = zipfile.ZipFile(p, 'r')
    zip_ref.extractall(dst)
    zip_ref.close()
            
def rename_tumor_directory(current_path):
    '''
    Handle an edge case: the liver tumor of the 7th patient is located in a directory with an inconsitent name.
    '''
    path = current_path + '3Dircadb1.7/MASKS_DICOM'
    print(path + '/tumor')
    if os.path.exists(path + '/tumor'):
        print('*** Dircadb1.7/MASK_DICOM/tumor is renamed ***')
        os.rename(path + '/tumor', path + '/livertumor')
    else:
        print('1.7 Tumor directory not renamed!')
        
def add_extension_dcm(path):
    '''
    Append .dcm extension to all file included in a directory.
    @param path The directory's path of patient CT scans slices.
    '''
    for filename in os.listdir(path):
        src = path + filename
        os.rename(src, src + '.dcm')
        
def get_patient_id(rep):
    '''
    Getter the patient id.
    '''
    return int(rep.split('.')[-1])

def get_image_id(img):
    '''
    Getter the image number.
    '''
    return int(img.split('.')[0].split('_')[-1])
        
class DataDCM():
    '''
    Data-Structure that eases bundling of CT-Scans with their respective segmentation.
    '''
    def __init__(self, size):
        self.X = {}
        self.Y = {}
        self.zeros = numpy.zeros((size,size))
        
    def add_x(self, filename, x):
        self.X[get_image_id(filename)] = x
        
    def add_y(self, filename, y, stack = False):
        id_image = get_image_id(filename)
        if id_image not in self.Y and stack == False:
            self.Y[id_image] = y
        elif stack == False:
            self.Y[id_image] += y
        else: # stack True (happens only if we set two_layers = True)
            T = self.Y[id_image] if (id_image in self.Y) else self.zeros
            self.Y[id_image] = numpy.stack((y, T), axis=-1)

    def get_data(self):
        data = []
        x_data, y_data = [], []
        blank_seg = False
        zeros = None
        if len(self.Y) == 0:
            print('The CT Scan has no found corresponding segmentation to the targeted mask. Blank segmentations will be set instead!!')
            blank_seg = True            
        elif len(self.X) != len(self.Y):
            print('ERROR: The two dict do not have the same size!\nThis should not happen!')
        keys = self.X.keys()
        keys.sort()
        for i in keys:
            if i in self.Y:
                data.append((self.X[i], self.Y[i]))
            else:
                if blank_seg:
                    data.append((self.X[i], self.zeros))
                else:
                    print('WARNING: indexes are not overlapping: '+ str(i))
        for d in data:
            x_data.append(d[0])
            y_data.append(d[1])
#        x_data = [x[0] for x in data]
#        y_data = [y[1] for y in data]
        return x_data, y_data
                

        
def from_dcm_to_png(path):
    '''
    Optional: Translates DICOM files to its visualization file in "png" format
    @param path The directory's path of patient CT scans slices.
    '''
    for filename in os.listdir(path):
        if filename.endswith('.dcm'):
            full_path = path + filename
            image = preprocess(full_path)
            imsave(full_path.replace("dcm","png"), image, "png")
            
               
def iterate_in_masks(full_masks_dir, mask, function):
    '''
    Carries out the passed function on multiple sub-directories, whose names start with mask's name (e.g. for mask = liver, "liver_tumor" is corresponds to a sub-directory on which the function will be performed. 
    @param full_masks_dir Path to MASKS_DICOM directory
    @param mask Chosen mask (e.g. liver, bones...)
    @param function Function that will be performed on all sub-directories, whose names start with @mask
    '''
    for masks_rep in os.listdir(full_masks_dir):
        if masks_rep.startswith(mask):
            function(full_masks_dir + masks_rep + '/')
            
                
def extract_and_add_dcm_ext(dirclab_dir ='./', extract_the_zip = True, mask= 'liver', with_tumors = True):
    '''
    Dezip all necessair data for each patient and append .dcm extension for all DICOM slices.
    @param dirclab_dir Path where all patients directories are located
    @param extract_the_zip "True" to unzip all patients "PATIENT_DICOM" & "MASKS_DICOM" files ("False" if they are already unzipped)
    @param mask The organ name that you want to unzip and add .dcm extension (and eventually to extends png images) (default: liver)
    @param with_tumor "True" to decompress / add extension / add images for tumors files (works for liver, to check for other organs)
    '''
    if with_tumors:
        print('Extracting tumors masks may extend the processing mode')
    for rep in os.listdir(dirclab_dir):
        if rep.startswith('3Dircadb1.') and os.path.isdir(dirclab_dir + rep):
            in_dircab = dirclab_dir + rep + '/'
            full_patient_dir = in_dircab + patient_dir 
            full_masks_dir = in_dircab + masks_dir 
            if extract_the_zip: 
                extract_zip(full_patient_dir, in_dircab)
                extract_zip(full_masks_dir, in_dircab)
                # handling an edge case
                if rep.endswith('.7'):
                    rename_tumor_directory(dirclab_dir)
            full_patient_dir = full_patient_dir + '/'
            full_masks_dir = full_masks_dir + '/'
            add_extension_dcm(full_patient_dir)
            if with_tumors:
                iterate_in_masks(full_masks_dir, mask, add_extension_dcm)
            else:
                add_extension_dcm(full_masks_dir + mask + '/')
                   
                
def is_dcm(img):
    return img.endswith('.dcm')
 
def flatten_dataset(dataset):
    '''flatten only the first dimension.
    @param dataset list of CT scans series of patients.
    @return numpy.array 
    '''
    flat = []
    for d in dataset:
        flat += d
    return numpy.array(flat)

def filter_directories(listdir,path, part):
    '''
    Helper for partitionning dataset into two parts.
    '''
    if part==0:
        return listdir
    splited= []
    if part==1:
        for d in listdir:
            if d.startswith('3Dircadb1.') and os.path.isdir(path + d):
                i = int(d.split('.')[1])
                if i<=7:
                    splited.append(d)
    if part==2:
        for d in listdir:
            if d.startswith('3Dircadb1.') and os.path.isdir(path + d):
                i = int(d.split('.')[1])
                if 7<i and i<=14:
                    splited.append(d)
                    
    if part==3:
        for d in listdir:
            if d.startswith('3Dircadb1.') and os.path.isdir(path + d):
                i = int(d.split('.')[1])
                if 14<i:
                    splited.append(d)
                    
    return splited        

def get_dataset(dirclab_dir ='./', mask = 'liver', resize_ratio = 1, tumor=False, two_layers =False, gaussian_noise=False, part=0):
    '''
    Getter of IRC dataset. 
    @param dirclab_dir Path to the dirclab directory. Set current path per default
    @param mask Dataset to get. Currently consistent for 'liver' segmentations.
    @param resize_ratio Ratio between 0 and 1 of the image resizement in terms of 512x512 images. (e.g. for resize_ratio=.5, image size yields 256x256)
    @param tumor (False per default), False to get complete organ segmentation. True to get "only" tumor segmentations.
    @param two_layers (False per default), False to get either mask or tumor segmentation. True to get both segmentations.
    @param gaussian_noise True to set gaussian noise on ct scans slices.
    @param part =0 to get data of all patients. 1= to get data of patients 1 to 10. 2= to get patients 11 to 20 data.
    @return data_x ct scans images. (remove flattening function to get ct scans series instead)
    @return data_y ct scans segmentations with respect to the given mask
    '''
    indexes = []
    data_x = []
    data_y = []
    n_of_series = 0
    for rep in filter_directories(os.listdir(dirclab_dir), dirclab_dir, part):
        if rep.startswith('3Dircadb1.') and os.path.isdir(dirclab_dir + rep):  
            if debug:
                print('Extract patient: %s'%rep.split('.')[1])
            in_dircab = dirclab_dir + rep + '/'
            full_patient_dir = in_dircab + patient_dir + '/'
            indexes.append(get_patient_id(rep))
            data_dcm = DataDCM(int(resize_ratio * 512))
            for input_name in os.listdir(full_patient_dir):
                if is_dcm(input_name):
                    data_dcm.add_x(input_name, preprocess(full_patient_dir + input_name, noise = gaussian_noise, resize_ratio = resize_ratio))
            full_masks_dir = in_dircab + masks_dir + '/'
            if two_layers or tumor: # get tumor segmentations
                for mask_dir in os.listdir(full_masks_dir):
                    if mask_dir.startswith(mask) and mask_dir != mask: # Tumor directories have other names than mask's
                        mask_dir_path = full_masks_dir + mask_dir + '/'
                        for target_name in os.listdir(mask_dir_path):
                            if is_dcm(target_name):
                                data_dcm.add_y(target_name, preprocess(mask_dir_path + target_name, resize_ratio = resize_ratio, segmentation= True))
            if two_layers or not tumor: # get liver segmentation
                full_masks_dir = full_masks_dir + mask + '/'
                for input_name in os.listdir(full_masks_dir):
                    if is_dcm(input_name):
                        data_dcm.add_y(input_name, preprocess(
                            full_masks_dir + input_name, resize_ratio = resize_ratio, segmentation = True), stack= two_layers)
            # X is a serie of ct scan for one patient
            X, Y = data_dcm.get_data()#
            X = numpy.array(X)
            data_x.append(X)
            data_y.append(Y)
            n_of_series += 1
    
    print('Number of series:' + str(n_of_series))
#    return standardize_serie(flatten_dataset(data_x)), flatten_dataset(data_y), indexes
    return data_x, data_y, indexes