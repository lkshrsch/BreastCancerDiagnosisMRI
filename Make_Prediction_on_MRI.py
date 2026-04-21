#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 16:47:06 2023

@author: deeperthought
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import sys

MODEL_WEIGHTS_PATH = '/model/pretrained_model_weights.npy'
            
T1_POST_PATH = "data/inputs/7A8rgvMkzi_20170511_091726/post.nii.gz"
DCE_IN_PATH = "data/inputs/7A8rgvMkzi_20170511_091726/slope1.nii.gz"
DCE_OUT_PATH = "data/inputs/7A8rgvMkzi_20170511_091726/slope2.nii.gz"            

T1_pre_nii_path = '' # If data already normalized, no need for T1pre image
MODALITY = 'axial' # 'axial'
SIDE = ''


if __name__ == '__main__':
    
    # Change directory
    project_directory = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(project_directory + '/code/')
    os.chdir(project_directory + '/')
    
    # Load util functions
    from utils import UNet_v0_2D_Classifier, load_and_preprocess, color_map, generate_gradCAM_image#, FocalLoss

    # Load model architecture
    model = UNet_v0_2D_Classifier(input_shape =  (512,512,3), pool_size=(2, 2), deconvolution=True, 
                                  depth=6, n_base_filters=42, activation_name="softmax", L2=1e-5, USE_CLINICAL=True)
    # Load pre-trained weights stored as numpy array to avoid tensorflow version incompatibility
    loaded_weights = np.load(project_directory + MODEL_WEIGHTS_PATH, allow_pickle=True, encoding='latin1')
    model.set_weights(loaded_weights) # no need to compile model as we are just doing inference.
    
    # Load and process data
    all_subject_channels = [T1_POST_PATH, 
                            DCE_IN_PATH, 
                            DCE_OUT_PATH] 
    
    X, shape = load_and_preprocess(all_subject_channels, T1_pre_nii_path=T1_pre_nii_path, side=SIDE, imaging_protocol=MODALITY, debug=False)
    
    
    # Clinical and demographic information if available
    MODE_CLINICAL = np.array([[0.  , 0.51, 0.  , 1.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 1.  ]])
          
    print('Data preprocessed.. model inference')    
    preds = model.predict([X, np.tile(MODE_CLINICAL, (shape[0], 1))], batch_size=1, verbose=1)[:,-1]

    # preds = model.predict([X, np.tile(MODE_CLINICAL, (shape[0], 1))], batch_size=1, use_multiprocessing=True, workers=10, verbose=0)[:,-1]
    print('prediction done..')
    
    global_prediction = np.max(preds)
    max_slice = np.argwhere(preds == global_prediction)[0][0]   
    
    # Generate gradCAM for in-slice visualization
    print('Generating gradCAM heatmap..')
    heatmap, img, superimposed_img = generate_gradCAM_image(model, X[max_slice:max_slice+1], MODE_CLINICAL ,alpha = 0.30)   
    # from skimage.transform import resize
    # heatmap_resized = resize(heatmap, output_shape=(512,512), anti_aliasing=True)
    
    # Plot results    
    rgb_color = color_map(global_prediction)

    axial_projection_t1post = np.max(X[:,:,:,0],2)
    axial_projection_slope1 = np.max(X[:,:,:,1],2)
    axial_projection_slope2 = np.max(X[:,:,:,2],2)
    
    plt.figure(1)
    fig, ax = plt.subplots(4,1, sharex=True, figsize=(5,10))    
    
    ax[0].set_title(f'{T1_POST_PATH.split("/")[-1]}\n P(cancer) = ' + str(np.round(global_prediction,3)) + f'\n Most suspicious slice: {max_slice}')
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
    
    plt.show()
    # plt.savefig(project_directory + '/figures/fig1.png', dpi=300)
       
    
    plt.figure(2)
    fig, ax = plt.subplots(1,4, figsize=(15,5))    
    
    ax[0].set_title(f'{T1_POST_PATH.split("/")[-1]}\n P(cancer) = '  + str(np.round(global_prediction,3)) + f'\n Most suspicious slice: {max_slice}')
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
    plt.show()
    
    # plt.savefig(project_directory + '/fig2.png', dpi=300)


    
    plt.figure(3)
    fig, axes = plt.subplots(1,3, figsize=(15,5))
    axes[0].imshow(np.rot90(X[max_slice:max_slice+1][0,:,:,0]), cmap='gray', aspect="auto"), axes[0].set_xticks([]) , axes[0].set_yticks([])
    axes[1].imshow(np.rot90(X[max_slice:max_slice+1][0,:,:,1]), cmap='gray', aspect="auto"), axes[1].set_xticks([]) , axes[1].set_yticks([])
    axes[2].imshow(np.rot90(superimposed_img), aspect="auto"); axes[2].set_title('T1post + heatmap'); axes[2].set_xticks([]) , axes[2].set_yticks([])
    plt.show()

    # plt.savefig(project_directory + '/fig3.png', dpi=300)
