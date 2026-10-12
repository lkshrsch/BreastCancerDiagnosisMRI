# Detection of breast cancer in MRI

This repository contains a pre-trained convolutional neural network for breast cancer detection in DCE-MRI. The provided pipeline takes as an input three volumetric MR images: A T1w post-contrast image, a DCE-in image and a DCE-out image (pre-processing for generating the difference images will be provided soon). A list of demographic and clinical information is optional (on the works). The output is a vector with the predicted probability of cancer for each sagittal slice in the input and a gradCAM image showing the location within the maximum probability slice:

![Results example](/figures/result_example-1.png)

# Attribution

If you use the code or weights in published work, please cite:

Hirsch, Lukas, Elizabeth J. Sutton, Yu Huang, Beliz Kayis, Mary Hughes, Danny Martinez, Hernan A. Makse, and Lucas C. Parra. "High-performance open-source Ai for breast cancer detection and localization in MRI." Radiology: Artificial Intelligence 7, no. 5 (2025): e240550.


# Installation

Code runs in the following tensorflow vesions (tested on colab): 
<br/>2.17.0*; 2.16.1*; 2.15.0*; 2.10.0; 2.9.0; 2.8.4; 2.0.0 
<br/>*recommended for faster inference.

google colab notebook provided for running demo on a public released axial MRI (see below Contents of project).

yaml file for installing a conda environment for tensorflow 2.0.0 provided

Docker container in the works...

# Run on Google Colab

To run a demo on the public dataset present, simply run the following lines on a google colab Jupiter notebook (make sure to change runtime type to give access to GPUs for faster inference).

<br/>`!git clone -b develop https://github.com/lkshrsch/BreastCancerDiagnosisMRI.git`
<br/>`%run BreastCancerDiagnosisMRI/Make_Prediction_on_MRI.py`


# Contents of project: 

This project is organized with the following directory structure:

**data/**

* This directory holds preprocessed MRI images in NIfTI format (`.nii.gz`) for analysis. Specific filenames suggest different acquisition sequences or orientations. This folder contains an axial MRI image of a breast. The image is obtained from a public dataset released by Duke
University (https://sites.duke.edu/mazurowski/resources/breast-cancer-mri-dataset/)

    * `T1_axial_02.nii.gz`
    * `T1_axial_slope1.nii.gz`
    * `T1_axial_slope2.nii.gz`


**model/**

* This directory stores the trained model for breast cancer diagnosis:
    * `pretrained_model_weights.npy` - This file contains the weights for a Convolutional Neural Network (CNN).

*Model Architecture*
The breast cancer detection model is a UNet architecture with the following specifications:

* Input shape: (512, 512, 3)
* Pooling size: (2, 2)
* Initial learning rate: 1e-5
* Depth: 6
* Base filters: 42
* Activation: Softmax
* L2 regularization: 1e-5
* Use of clinical data: True

*Model Weights*
The pre-trained weights for this model are stored in `CNN_weights.npy`. These weights can be loaded into the model using the following code:

```python
import numpy as np
from model_utils import FocalLoss, UNet_v0_2D_Classifier

model_weights = np.load('/path/to/your/model/CNN_weights.npy', allow_pickle=True)

model = UNet_v0_2D_Classifier(input_shape=(512, 512, 3), pool_size=(2, 2), initial_learning_rate=1e-5,
                             deconvolution=True, depth=6, n_base_filters=42,
                             activation_name="softmax", L2=1e-5, USE_CLINICAL=True)
model.set_weights(model_weights)
``` 


**code/**

* This directory holds scripts for various tasks:
    * `model_utils.py` - This script contain utility functions for working with the model.
    * `make_diagnosis_on_MRI.py` - This script is used to make predictions on unseen MRI scans.
    * `train_model.py` - This script is used to train the CNN model from scratch or fine-tune existing weights.

**training_data/**

* This directory holds training data used to train the model:
    * `Data_Description.csv` - The specific data here was generated at random and only serves the purpose of demonstrating the expected data format:

      * DE-ID: De-identified patient identifier. In this case just a number
      * Exam: Identifier for exam when image was taken
      * Scan_ID : Identifier of a specific scan: Constructed from DE-ID + Exam + side (in case of a saggital image)
      * Partition : Describes whether image is part of the training or validation set
      * Image : Points to location of the specific file containing the image of a scan. Name is constructed using DE-ID + Exam + Scan_ID + slice number of image.
      * BIRADS : From clinical information of exam
      * Pathology : From clinical information of exam 
      * Age : From clinical information of exam
      * Family Hx : From clinical information of exam
      * Ethnicity and Race : From clinical information of exam

**sessions/**

* This folder is used to store the output of each new training session. It saves the trained
models, evaluation metrics, and any other relevant information generated during the
training process.

# Details of the pre-trained model

**Training Data**
This network was trained on sagittal breast MRI taken at a tertiary cancer center between 2002-2014. The data included 38,005 exams (31,564 screening, 6,015 diagnostic, 426 unknown or N/A) from 12,329 patients. Counting each breast individually yielded 65,105 sagittal breast images with 2,690 malignant images. This data was randomly divided by patient into training, validation, and test sets (90/10 for training and test, and subsequently 90/10 for training and validation patients). This dataset included radiologist annotations/segmentation on the slice containing the largest (index) cancer for all 2,690 malignant breast images (termed the "index slice"). 

**Data pre-processing and harmonization**
Preprocessing followed our previous work in segmentation. Briefly, pre- and post-contrast T1-weighted images were co-registered using NiftyReg, and dynamic contrast enhancement was summarized into images capturing initial contrast uptake (DC-in) and washout (DC-out), alongside the first post-contrast T1-weighted image (T1-post). These three channels were normalized by dividing by the 95th percentile of the pre-contrast T1-weighted image in each exam. To adjust for inter-channel differences, each channel was divided by its 95th percentile across the training set. This resulted in factors of 40 for T1-post, 0.3 for DCE-in, and 0.12 for DCE-out, applied to both sagittal and axial MRI. The sagittal MRI data had varying in-plane resolutions (0.4mm to 0.8mm). Low-resolution images were upsampled by a factor of two for harmonization. Axial images were resampled to match a sagittal in-plane resolution of 0.4mm and separated into left and right breasts. All images were cropped to 512x512 pixels, ensuring the breast was centered.

**Training**
The model is trained using index slices from malignant images as positive examples. As negative examples, we selected the center slice and one randomly selected slice from benign images. All models were trained using a focal loss with alpha=5, using the ”Adam” optimizer with learning rate=1e-5. All models were trained for 100 epochs, and the model with the lowest validation loss was saved for evaluation. Unless specified, all models were trained with data augmentation, consisting of random rotation within 60 degrees, random shear of scale 0.1, random horizontal and vertical flips, and random intensity scaling in range 0.8-1.2, all implemented using the TensorFlow preprocessing ImageDataGenerator library.

# License and Disclaimer

This repository contains code and trained model weights. The code is licensed under the Apache License 2.0, found in the file LICENSE. The model weights are licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0), found in the file LICENSE-WEIGHTS. The weights may be used and adapted for noncommercial purposes only, and adapted weights, including fine-tuned weights, must be shared under the same license.

The model is a research tool. It has not been cleared or approved by the FDA or any other regulatory body, and it has not been validated for clinical use. Do not use it to diagnose, treat, or manage any patient, or as a substitute for the judgment of a qualified clinician. The model can produce incorrect results, and its performance may be worse on images from
other scanners, sites, or patient populations.

The code and weights are provided "as is", without warranty of any kind, express or implied. To the extent permitted by law, the authors accept no liability for any damages arising from the use of the code, the weights, or the results they produce.
