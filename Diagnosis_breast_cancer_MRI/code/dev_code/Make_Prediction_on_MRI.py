#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 16:47:06 2023

@author: deeperthought
"""

import os
os.chdir('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/')
from utils import UNet_v0_2D_Classifier, load_and_preprocess, color_map, generate_gradCAM_image

import numpy as np
import matplotlib.pyplot as plt

#%%

MODEL_WEIGHTS_PATH = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/best_model_weights.npy'

#---- MSKCC Sagittal Exam -----------------------
# t1post_path = "/media/HDD/DATA/MRI/MSKCC_16-328_1_03852_20111214/T1_right_02_01.nii"
# slope1_path = "/media/HDD/DATA/MRI/MSKCC_16-328_1_03852_20111214/T1_right_slope1.nii"
# slope2_path = "/media/HDD/DATA/MRI/MSKCC_16-328_1_03852_20111214/T1_right_slope2.nii"
# T1_pre_nii_path = '/media/HDD/DATA/MRI/MSKCC_16-328_1_03852_20111214/T1_right_01_01.nii'
# MODALITY = 'sagittal' # 'axial'
# SIDE = 'right'

#---- MSKCC Sagittal Exam -----------------------
# t1post_path = "/media/HDD/DATA/MRI/MSKCC_16-328_1_09291_20060427/T1_left_02_01.nii"
# slope1_path = "/media/HDD/DATA/MRI/MSKCC_16-328_1_09291_20060427/T1_left_slope1.nii"
# slope2_path = "/media/HDD/DATA/MRI/MSKCC_16-328_1_09291_20060427/T1_left_slope2.nii"
# T1_pre_nii_path = '/media/HDD/DATA/MRI/MSKCC_16-328_1_09291_20060427/T1_left_01_01.nii'
# MODALITY = 'sagittal' # 'axial'
# SIDE = 'left'


# DATA_PATH = '/home/deeperthought/kirbyPRO/alignedNiiAxial-May2020-cropped-normed/'
# EXAM = 'MSKCC_16-328_1_00001_20150710' # benign
# EXAM = 'MSKCC_16-328_1_02914_20140516' # malignant right

#---- MASKCC Axial Exam -----------------------
t1post_path = "/home/deeperthought/kirbyPRO/alignedNiiAxial-May2020-cropped-normed/MSKCC_16-328_1_02914_20140516/T1_axial_02_01.nii.gz"
slope1_path = "/home/deeperthought/kirbyPRO/alignedNiiAxial-May2020-cropped-normed/MSKCC_16-328_1_02914_20140516/T1_axial_slope1.nii.gz"
slope2_path = "/home/deeperthought/kirbyPRO/alignedNiiAxial-May2020-cropped-normed/MSKCC_16-328_1_02914_20140516/T1_axial_slope2.nii.gz"
T1_pre_nii_path = ''
MODALITY = 'axial' # 'axial'
SIDE = ''


# #---- JHU Exam -----------------------
# t1post_path = "/home/deeperthought/kirbyPRO/jhuData/alignedNii-normed/ML00925838_20140504/T1_axial_02.nii"
# slope1_path = "/home/deeperthought/kirbyPRO/jhuData/alignedNii-normed/ML00925838_20140504/T1_axial_slope1.nii"
# slope2_path = "/home/deeperthought/kirbyPRO/jhuData/alignedNii-normed/ML00925838_20140504/T1_axial_slope2.nii"
# T1_pre_nii_path = ''
# MODALITY = 'axial' # 'axial'
# SIDE = ''


#%%
model = UNet_v0_2D_Classifier(input_shape =  (512,512,3), pool_size=(2, 2),initial_learning_rate=1e-5, 
                                         deconvolution=True, depth=6, n_base_filters=42,
                                         activation_name="softmax", L2=1e-5, USE_CLINICAL=True)

loaded_weights = np.load(MODEL_WEIGHTS_PATH, allow_pickle=True, encoding='latin1')

model.set_weights(loaded_weights)


#%%

all_subject_channels = [t1post_path, slope1_path, slope2_path]
 
X, shape = load_and_preprocess(all_subject_channels, T1_pre_nii_path=T1_pre_nii_path, side=SIDE, imaging_protocol=MODALITY, debug=True)

print('Data preprocessed.. model inference')


MODE_CLINICAL = np.array([[0.  , 0.51, 0.  , 1.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 1.  ]])
# clinical['exam'] = clinical['scan_ID'].str[:-2]
# clinical = clinical.drop_duplicates('exam')
# clinical = pd.read_csv('/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/FullData_RandomSlices_DataAug__classifier_train52598_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/Clinical_Data_Train.csv')
# if EXAM in clinical['exam'].values:
#     clinical_features = clinical.loc[clinical['exam'] == EXAM, [u'Family Hx',u'Age',u'ETHNICITY_HISPANIC OR LATINO',u'ETHNICITY_NOT HISPANIC', u'ETHNICITY_UNKNOWN',u'RACE_ASIAN-FAR EAST/INDIAN SUBCONT',u'RACE_BLACK OR AFRICAN AMERICAN',u'RACE_NATIVE AMERICAN-AM IND/ALASKA',u'RACE_NATIVE HAWAIIAN OR PACIFIC ISL',u'RACE_UNKNOWN',u'RACE_WHITE']].values
# else:
#     clinical_features = np.array(MODE_CLINICAL)
    
    
preds = model.predict([X, np.tile(MODE_CLINICAL, (shape[0], 1))], batch_size=1, use_multiprocessing=True, workers=10, verbose=0)[:,-1]
    
print('prediction done..')
# preds_gated = preds*slices_on_breast[1:]
global_prediction = np.max(preds)
max_slice = np.argwhere(preds == global_prediction)[0][0]

if MODALITY == 'axial':
    image_half = preds.shape[0]//2
    left_breast_predictions = preds[:image_half]
    right_breast_predictions = preds[image_half:]
    print(f'{image_half}')
    print(f'{len(left_breast_predictions)}, {len(right_breast_predictions)}')
        
    left_breast_global_pred = np.max(left_breast_predictions)
    right_breast_global_pred = np.max(right_breast_predictions)


axial_projection_t1post = np.max(X[:,:,:,0],2)
axial_projection_slope1 = np.max(X[:,:,:,1],2)
axial_projection_slope2 = np.max(X[:,:,:,2],2)

print('Generating gradCAM heatmap..')
heatmap, img, superimposed_img = generate_gradCAM_image(model, X[max_slice:max_slice+1], MODE_CLINICAL ,alpha = 0.30)


# from skimage.transform import resize
# heatmap_resized = resize(heatmap, output_shape=(512,512), anti_aliasing=True)


#%%

rgb_color = color_map(global_prediction)

plt.figure(1)
fig, ax = plt.subplots(4,1, sharex=True, figsize=(5,10))    

ax[0].set_title(f'{t1post_path}\n P(cancer) = ' + str(np.round(global_prediction,3)) + f'\n Most suspicious slice: {max_slice}')
ax[0].plot(preds)
ax[0].vlines(max_slice,0,global_prediction,color=rgb_color)
ax[0].set_aspect('auto')

ax[1].imshow(np.rot90(axial_projection_t1post), cmap='gray')   
ax[1].set_aspect('auto'); ax[1].set_xticks([]); ax[1].set_yticks([])
ax[1].vlines(max_slice,0,512,color=rgb_color)
ax[1].set_aspect('auto');
   

ax[2].imshow(np.rot90(axial_projection_slope1), cmap='gray')   
ax[2].set_aspect('auto'); ax[1].set_xticks([]); ax[2].set_yticks([])
ax[2].vlines(max_slice,0,512,color=rgb_color)
ax[2].set_aspect('auto');
   

ax[3].imshow(np.rot90(axial_projection_slope2), cmap='gray')   
ax[3].set_aspect('auto'); ax[1].set_xticks([]); ax[3].set_yticks([])
ax[3].vlines(max_slice,0,512,color=rgb_color)
ax[3].set_aspect('auto');
   

plt.figure(2)
fig, ax = plt.subplots(1,4, figsize=(15,5))    

ax[0].set_title(f'{t1post_path}\n P(cancer) = ' + str(np.round(global_prediction,3)) + f'\n Most suspicious slice: {max_slice}')
ax[0].plot(preds)
ax[0].vlines(max_slice,0,global_prediction,color=rgb_color)
ax[0].set_aspect('auto')
ax[0].set_xlabel('Sagittal slice number')
# ax[0].set_xticks(np.arange(len(preds)))

ax[1].set_title('T1post')
ax[1].imshow(np.rot90(X[max_slice,:,:,0]), cmap='gray' )
ax[1].set_xlabel(f'Predicted slice: {max_slice}'); ax[1].set_xticks([]); ax[1].set_yticks([])
ax[1].set_aspect('auto')
for spine in ax[1].spines.values():
    spine.set_edgecolor(rgb_color)  # Change color as desired
    spine.set_linewidth(2)  # Change thickness as desired


ax[2].set_title('DCE-in')
ax[2].imshow(np.rot90(X[max_slice,:,:,1]), cmap='gray' )
ax[2].set_xlabel(f'Predicted slice: {max_slice}'); ax[2].set_xticks([]); ax[2].set_yticks([])
ax[2].set_aspect('auto')
for spine in ax[2].spines.values():
    spine.set_edgecolor(rgb_color)  # Change color as desired
    spine.set_linewidth(2)  # Change thickness as desired

ax[3].set_title('DCE-out')
ax[3].imshow(np.rot90(X[max_slice,:,:,2]), cmap='gray' )
ax[3].set_xlabel(f'Predicted slice: {max_slice}'); ax[3].set_xticks([]); ax[3].set_yticks([])
ax[3].set_aspect('auto')
for spine in ax[3].spines.values():
    spine.set_edgecolor(rgb_color)  # Change color as desired
    spine.set_linewidth(2)  # Change thickness as desired

plt.tight_layout()


plt.figure(3)
fig, axes = plt.subplots(1,3, figsize=(15,5))
axes[0].imshow(np.rot90(X[max_slice:max_slice+1][0,:,:,0]), cmap='gray', aspect="auto"), axes[0].set_xticks([]) , axes[0].set_yticks([])
axes[1].imshow(np.rot90(X[max_slice:max_slice+1][0,:,:,1]), cmap='gray', aspect="auto"), axes[1].set_xticks([]) , axes[1].set_yticks([])
axes[2].imshow(np.rot90(superimposed_img), aspect="auto"); axes[2].set_title('T1post + heatmap'); axes[2].set_xticks([]) , axes[2].set_yticks([])