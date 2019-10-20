# Official Master Thesis

This repository includes the necessary implementation to reproduce the experiments highlighted in the master thesis.

## Necessary dependencies
* Python@2.7 - 3.6
* Tensorflow-gpu@1.5
* Keras
* numpy
* Nibabel
* Scipy
* medpy

## Datasets
The LiTS dataset includes 130 CT-Scans series in .nii format with their respective liver/lesions segmentations classified 1 and 2 respectively (while 0 for background).

The dataset is represented in 3-d format in sagittal cut that will be translated to axial cut in the learning task.

An example of the ct-scans visualization overlayed with the groundtruth segmentation is located in "data_example".

This thesis comprises the use of D3-ircad database that includes 20 CT volumes for patients. 15 out of 20 scans feature tumors in liver.

The data is represented in DICOM format. This project includes the necessary utility to translate the downloaded database to a consumable .h5 database. 

NOTICE: This repository includes no dataset. Please consult the following sites to learn more:

* https://competitions.codalab.org/competitions/17094

* https://www.ircad.fr/research/3d-ircadb-01/
