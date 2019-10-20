import numpy as np

def logarithmic_cropping(seg, c= 48, second = None, third = None, offset = None, previous = None, tolerance = 20):
'''Computes the cropping cordinates that maximize the size of the liver in the mask.
   @param seg Mask
   @param c The size to reduce for each axis
   @param tolerance Used to stop the recursion if the cropped liver loses less than "tolerance" liver pixels
   @return cropped segmentation
   @return cropping coordinates in the X axis
   @return cropping coordinates in the Y axis
'''
    cropped = previous if previous is not None else seg[:,c:-c, c:-c]
    s = seg.sum()
    s_c = cropped.sum()
    
    if (s - s_c) < tolerance:
        return cropped, (c, -c), (c, -c)
    
    t = offset if offset is not None else int(c/2)
    
    if second is None and third is None:
        a_1, a_2, a_3, a_4 = c, -c, c, -c
    else: 
        a_1, a_2, a_3, a_4 = second[0], second[1], third[0], third[1]
    # top
    cropped_0 = seg[:, a_1 + t : a_2 + t, c : -c]
    # down
    cropped_1 = seg[:, a_1 - t : a_2 - t, c : -c] 
    # left
    cropped_2 = seg[:, c : -c, a_3 + t : a_4 + t]
    # right
    cropped_3 = seg[:, c : -c, a_3 - t : a_4 - t]
    
    # top - left
    cropped_4 = seg[:, a_1 + t : a_2 + t, a_3 + t : a_4 + t]
    # down - left
    cropped_5 = seg[:, a_1 - t : a_2 - t, a_3 + t : a_4 + t] 
    # top - right 
    cropped_6 = seg[:, a_1 + t : a_2 + t, a_3 - t : a_4 - t]
    # down - right
    cropped_7 = seg[:, a_1 - t : a_2 - t, a_3 - t : a_4 - t]
    
    sums = [cropped_0.sum(), cropped_1.sum(), cropped_2.sum(), cropped_3.sum(),
            cropped_4.sum(), cropped_5.sum(), cropped_6.sum(), cropped_7.sum()]

    if t == 1:
        print 'Coverage: '+ str(float(s_c)/float(s))
    if max(sums) < s_c:
        if t == 1:
            return previous, (a_1, a_2), (a_3, a_4)
        else:
            return logarithmic_cropping(seg, c, (a_1, a_2), (a_3, a_4), int(t/2), previous)
    
    perfect = (s - max(sums)) <  tolerance
    index_min = np.argmax(sums)
    if index_min == 0:
        if perfect or t == 1:
            return cropped_0, (a_1 + t, a_2 + t), (c, -c)
        else:
            return logarithmic_cropping(seg, c, (a_1 + t, a_2 + t), (c, -c), int(t/2), cropped_0)
    elif index_min == 1:
        if perfect or t == 1:
            return cropped_1, (a_1 - t, a_2 - t), (c, -c)
        else:
            return logarithmic_cropping(seg, c, (a_1 - t, a_2 - t), (c, -c), int(t/2), cropped_1)
    elif index_min == 2:
        if perfect or t == 1:
            return cropped_2, (c, -c), (a_3 + t, a_4 + t) 
        else:
            return logarithmic_cropping(seg, c, (c, -c), (a_3 + t, a_4 + t), int(t/2), cropped_2)
    elif index_min == 3:
        if perfect or t == 1:
            return cropped_3, (c, -c), (a_3 - t, a_4 - t)
        else:
            return logarithmic_cropping(seg, c, (c, -c), (a_3 - t, a_4 - t), int(t/2), cropped_3)
    elif index_min == 4:
        if perfect or t == 1:
            return cropped_4, (a_1 + t, a_2 + t), (a_3 + t, a_4 + t)
        else:
            return logarithmic_cropping(seg, c, (a_1 + t, a_2 + t), (a_3 + t, a_4 + t), int(t/2), cropped_4)
    elif index_min == 5:
        if perfect or t == 1:
            return cropped_5, (a_1 - t, a_2 - t), (a_3 + t, a_4 + t)
        else:
            return logarithmic_cropping(seg, c, (a_1 - t, a_2 - t), (a_3 + t, a_4 + t), int(t/2), cropped_5)
    elif index_min == 6:
        if perfect or t == 1:
            return cropped_6, (a_1 + t, a_2 + t), (a_3 - t, a_4 - t)
        else:
            return logarithmic_cropping(seg, c, (a_1 + t, a_2 + t), (a_3 - t, a_4 - t), int(t/2), cropped_6)
    elif index_min == 7:
        if perfect or t == 1:
            return cropped_7, (a_1 - t, a_2 - t), (a_3 - t, a_4 - t)
        else:
            return logarithmic_cropping(seg, c, (a_1 - t, a_2 - t), (a_3 - t, a_4 - t), int(t/2), cropped_7)

    raise 'Should not happen!'

def crop_x_y(x,y):
'''Reduces the image dimension length by 48 pixels
@param x  volume
@param y  segmentation
@return x cropped volume
@return y cropped segmentation
'''
y, a, b = logarithmic_cropping(y, c=48, tolerance= 20)
x = x[:, a[0]:a[1], b[0]:b[1]]
return x, y

def normalize_data(data):
    '''Normalize each slice in the volume'''
    for i in range(data.shape[0]):
        # data[i] = image_histogram_equalization(data[i])
        std = np.std(data[i])
        data[i] -= np.mean(data[i])
        if std != 0:
            data[i] /= std
    return data

def image_histogram_equalization(image, number_bins=2047):
    '''Histogram equalizer'''
    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape)