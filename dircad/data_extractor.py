import os, zipfile, numpy, dicom
from scipy.misc import imsave
from scipy.ndimage.interpolation import zoom
from scipy.misc import imsave, imshow


patient_dir = 'PATIENT_DICOM'
masks_dir = 'MASKS_DICOM'

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
        else:
            self.Y[id_image] += y

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
                    print('WARNING: indices are not overlapping: '+ str(i))
        for d in data:
            x_data.append(d[0])
            y_data.append(d[1])
#        x_data = [x[0] for x in data]
#        y_data = [y[1] for y in data]
        return x_data, y_data
           
               
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
            
                
def extract_and_add_dcm_ext(dirclab_dir ='./', extract_the_zip = True, with_tumors = True):
    '''
    Dezip all necessair data for each patient and append .dcm extension for all DICOM slices.
    @param dirclab_dir Path where all patients directories are located
    @param extract_the_zip "True" to unzip all patients "PATIENT_DICOM" & "MASKS_DICOM" files ("False" if they are already unzipped)
    @param mask The organ name that you want to unzip and add .dcm extension (and eventually to extends png images) (default: liver)
    @param with_tumor "True" to decompress / add extension / add images for tumors files (works for liver, to check for other organs)
    '''
    
    mask = 'liver'
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
                # handling patient 7 features a special case
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

def threshold_segmentations(y, sig = .50):
    y[y >= sig] = 1
    y[y < sig] = 0
    return y

def get_slice(filename, resize_ratio=1, noise = False, segmentation = False):
    # in this tutorial we resize the slice to 512/4 = 128x128
    img = dicom.read_file(filename).pixel_array
    if segmentation: # Some masks are not binary, this line maps the segmentation masks to a binary format
        img = threshold_segmentations(img)
    if resize_ratio < 1:
        img = zoom(img, resize_ratio)
        if segmentation:
            img = threshold_segmentations(img)            

    return img

def get_patient_case_dirc(cid, dirclab_dir ='./', resize_ratio = 1):
    '''
    Getter of IRCAD patient data. The function assumes that the data is already extracted by the "extract_and_add_dcm_ext" extractor.
    @param cid The id of the patient. Ranging in [1, 20].
    @param dirclab_dir Path to the dirclab directory. Set current path per default
    @param tumor (False per default), False to get complete organ segmentation. True to get "only" tumor segmentations.
    @return X ct scans images. (remove flattening function to get ct scans series instead)
    @return Y ct scans segmentations with respect to the given mask
    '''
    mask = 'liver'
    for rep in os.listdir(dirclab_dir):
        if rep == '3Dircadb1.'+str(cid) and os.path.isdir(dirclab_dir + rep):  
            print('Extracting patient %s ...'%rep.split('.')[1])
            in_dircab = dirclab_dir + rep + '/'
            full_patient_dir = in_dircab + patient_dir + '/'
            data_dcm = DataDCM(int(resize_ratio * 512))
            for input_name in os.listdir(full_patient_dir):
                if is_dcm(input_name):
                    data_dcm.add_x(input_name, get_slice(full_patient_dir + input_name, resize_ratio = resize_ratio))
            full_masks_dir = in_dircab + masks_dir + '/'
            # Get tumor masks
            for mask_dir in os.listdir(full_masks_dir):
                if mask_dir.startswith(mask) and mask_dir != mask: # Tumor directories have other names than mask's
                    mask_dir_path = full_masks_dir + mask_dir + '/'
                    for target_name in os.listdir(mask_dir_path):
                        if is_dcm(target_name):
                            data_dcm.add_y(target_name, get_slice(mask_dir_path + target_name, resize_ratio = resize_ratio, segmentation = True))
            # Get liver masks
            full_masks_dir = full_masks_dir + mask + '/'
            for input_name in os.listdir(full_masks_dir):
                if is_dcm(input_name):
                    data_dcm.add_y(input_name, get_slice(full_masks_dir + input_name, resize_ratio = resize_ratio, segmentation = True))
            # X is a serie of ct scan for one patient
            X, Y = data_dcm.get_data()#
            return numpy.array(X), numpy.array(Y).astype('uint8')
    print(dirclab_dir + '3Dircadb1.'+str(cid)+ ' not found!')
    return None, None