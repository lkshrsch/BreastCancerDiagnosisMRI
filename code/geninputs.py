import os
import pydicom as pyd
import glob
import numpy as np
import pandas as pd
import nibabel as nib
from typing import Callable, List, Any
from multiprocessing import Queue, Manager, cpu_count
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import subprocess
import threading
from toolbox import ProgressBar, get_logger
# Global variables for progress bar and lock
Progress = None
manager = Manager()
progress_queue = manager.Queue()
LOGGER = get_logger('06_genInputs', 'data/logs/')

SAVE_DIR = '/scratch/joyliu/data/breast_mr_ccny/'
DEBUG = 0
TEST = True
N_TEST = 40
PARALLAL = True
PROGRESS = False
# This script is for generating the numpy files utilized for model training
# Performs the calculation of the slope 1 (enhancement) for each scan
# Performs the calculation of the slope 2 (washout) for each scan
# Normalizes samples by dividing by 95th percentile of T1_01_01
def progress_wrapper(item, target, progress_queue, *args, **kwargs):
    result = target(item, *args, **kwargs)
    progress_queue.put((None, f'Processing'))
    return result

def run_with_progress(target: Callable[..., Any], items: List[Any], Parallel: bool=True, *args, **kwargs) -> List[Any]:
    """Run a function with a progress bar"""
    # Initialize using a manager to allow for shared progress queue
    manager = Manager()
    progress_queue = manager.Queue()
    target_name = target.func.__name__ if isinstance(target, partial) else target.__name__

    # Debugging information
    LOGGER.debug(f'Running {target_name} with progress bar')
    LOGGER.debug(f'Number of items: {len(items)}')
    LOGGER.debug(f'Parallel: {Parallel}')

    # Initialize progress bar
    if PROGRESS:
        Progress = ProgressBar(len(items))
        updater_thread = threading.Thread(target=progress_updater, args=(progress_queue, Progress))
        updater_thread.start()
    
    # Pass the progress queue to the target function
    target = partial(progress_wrapper, target=target, progress_queue=progress_queue, *args, **kwargs)

    # Run the target function with a progress bar
    if Parallel:
        with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
            futures = [executor.submit(target, item, *args, **kwargs) for item in items]
            results = [future.result() for future in futures]
    else:
        results = [target(item) for item in items]

    # Close the progress bar
    if PROGRESS:
        progress_queue.put(None)
        print('\n')
        updater_thread.join()

    LOGGER.debug(f'Completed {target_name} with progress bar')
    LOGGER.debug(f'Number of results: {len(results)}')

    # Check if results is a list of tuples before returning zip(*results)
    if results and isinstance(results[0], tuple):
        return zip(*results)
    return results

def progress_updater(queue, progress_bar):
    while True:
        item = queue.get()
        if item is None:
            break
        index, status = item
        progress_bar.update(index, status)

        queue.task_done()

from datetime import datetime

def hms_to_s(hms):
    hms = (hms or '').split('.')[0].zfill(6)
    return int(hms[0:2])*3600 + int(hms[2:4])*60 + int(hms[4:6])

def estimate_phase_times_seconds(df):
    # df: rows for one SessionID, sorted in acquisition order
    # expects columns: AcqDate (YYYYMMDD), AcqTime (HH:MM:SS or HHMMSS), ScanDur (µs or 'Unknown')
    starts = []
    mids = []
    for _, r in df.iterrows():
        acq_time = (r['AcqTime'] or '').replace(':','')
        start_s = hms_to_s(acq_time)
        dur_s = float(r['ScanDur'])/1e6 if isinstance(r['ScanDur'], (int, float)) else None
        mid_s = start_s + (dur_s/2.0 if dur_s else 0.0)
        starts.append(start_s)
        mids.append(mid_s)
    t0 = mids[0]
    return [max(0.0, m - t0) for m in mids]  # seconds relative to pre

    
def generate_slopes(SessionID):
    # Should generate 2 slopes
    # Slope 1 - between 00 and 01
    # Slope 2 - between 01 and 0X
    LOGGER.warning(f'Generating slopes for session: {SessionID}')
    
    # Select timing rows for this session and sort by Major (acquisition order)
    Data = Data_table[Data_table['SessionID'] == SessionID]
    if len(Data) == 0:
        LOGGER.warning(f'{SessionID} | Skipping: no timing rows found')
        return

    # Determine dataset directory from sample_name and read NIfTI files (read-only)
    dataset_base = '/mnt/shareddata/datasets/breast_ucsf_mri/contrast_pixel_space/data'
    sample_vals = Data['sample_name'].dropna().unique().tolist() if 'sample_name' in Data.columns else []
    if len(sample_vals) == 0:
        LOGGER.warning(f"{SessionID} | Skipping: no sample_name in timing table for session")
        return
    sample_name = str(sample_vals[0])
    if len(sample_vals) > 1:
        LOGGER.warning(f"{SessionID} | Multiple sample_name values detected: {sample_vals}. Using {sample_name}.")

    dataset_dir = f'{dataset_base}/{sample_name}'
    if not os.path.exists(dataset_dir):
        LOGGER.warning(f"{SessionID} | Skipping: dataset directory not found: {dataset_dir}")
        return
    Fils = glob.glob(f'{dataset_dir}/*.nii.gz')
    Fils.sort()
    if len(Fils) == 0:
        LOGGER.warning(f"{SessionID} | Skipping: no NIfTI files found in {dataset_dir}")
        return
    LOGGER.warning(f'{SessionID} | Using dataset dir: {dataset_dir} | Files: {len(Fils)}')
    
    # Define output directory by sample_name, not SessionID
    output_dir = f"{SAVE_DIR}/{sample_name}"
    
    Major = Data['Major'] # Major is the order of the scans
    sorting = np.argsort(Major) # Sorting the scans
    LOGGER.debug(f'{SessionID} | Sorting values| {sorting.values}')
    LOGGER.debug(f'{SessionID} | Trigger Time | {Data["TriTime"].values}')
    LOGGER.debug(f'{SessionID} | Scan Duration | {Data["ScanDur"].values}')
    
    # Compose phase times in seconds
    Data_sorted = Data.iloc[sorting].reset_index(drop=True)

    # Align files to timing rows using Series token present in filenames
    files_upper = [(f, os.path.basename(f).upper()) for f in Fils]
    used = set()
    matched_files = []
    matched_idx = []

    def norm_series_token(s):
        s = str(s or '').upper().replace(' ', '')
        # Normalize some common variants
        s = s.replace('PHASE', 'PH')
        return s

    for idx, row in Data_sorted.iterrows():
        token = norm_series_token(row.get('Series', ''))
        found = None
        if token:
            for f, name_up in files_upper:
                if f in used:
                    continue
                if token in name_up:
                    found = f
                    break
        if found is None:
            # No direct token match; attempt fallback by looking for SeriesNumber pattern
            series_num = row.get('Orig Series #') if 'Orig Series #' in Data_sorted.columns else row.get('SeriesNumber', None)
            if series_num is not None:
                series_num_str = str(int(series_num)) if isinstance(series_num, (int, float)) else str(series_num)
                for f, name_up in files_upper:
                    if f in used:
                        continue
                    if series_num_str in name_up:
                        found = f
                        break
        if found is not None:
            used.add(found)
            matched_files.append(found)
            matched_idx.append(idx)

    if len(matched_files) < 3:
        LOGGER.debug(f"{SessionID} | Skipping: only matched {len(matched_files)} series to timing rows (need >=3).")
        return

    if len(matched_files) < len(Data_sorted):
        LOGGER.debug(f"{SessionID} | {len(matched_files)}/{len(Data_sorted)} timing rows matched to nifti files; continuing with matched subset.")

    # Filter timing rows to those with matched files and reset index
    Data_sorted = Data_sorted.iloc[matched_idx].reset_index(drop=True)
    Fils = matched_files
    post_tris = Data_sorted['TriTime'][1:].values
    # If any post scan has unknown TriTime, fallback to estimating from AcqTime/ScanDur
    LOGGER.debug(f"{SessionID} | Post-scan TriTime unknown. Estimating phase times from AcqTime/ScanDur.")
    try:
        Times = estimate_phase_times_seconds(Data_sorted)
    except Exception as e:
        LOGGER.error(f'{SessionID} | Failed to estimate phase times from AcqTime/ScanDur')
        LOGGER.error(f'{SessionID} | {e}')
        return
    
            
    LOGGER.debug(f'{SessionID} | Times | {Times}')
    
    # Load the 01 scan
    img = nib.load(Fils[0])
    data0 = img.get_fdata()
    data0[np.isnan(data0)] = 0
    p95 = float(np.percentile(data0,95))
    LOGGER.debug(f'{SessionID} | 95% | {p95}')

    header = img.header.copy()
    header['datatype'] = 16 # 32-bit float
    header['scl_slope'] = 1
    header['bitpix'] = 32
    header['cal_max'] = 0
    header['cal_min'] = 0
    
    # Create a new NIfTI image with the same affine, but with the data type, slope, and intercept set explicitly
    #new_img = nib.Nifti1Image(data0, img.affine)
    #new_img.header['datatype'] = 16
    #new_img.header['scl_slope'] = 1
    #new_img.header['bitpix'] = 32
    #new_img.header['cal_max'] = 0
    #new_img.header['cal_min'] = 0

    # Building time matrix same shape as loaded data
    T = np.zeros_like(data0, dtype=np.float32)
    T = np.expand_dims(T, axis=-1)
    T = np.repeat(T, len(Times), axis=-1)
    for ii,jj in enumerate(Times):
        T[:,:,:,ii] = jj
    
    # Loading all image data into single matrix
    D = np.zeros_like(data0, dtype=np.float32)
    D = np.expand_dims(D, axis=-1)
    D = np.repeat(D, len(Times), axis=-1)
    for ii,jj in enumerate(Fils):
        img = nib.load(jj)
        data0 = img.get_fdata().astype(np.float32)
        data0[np.isnan(data0)] = 0
        D[:,:,:,ii] = data0
    D[np.isnan(D)] = 0

    LOGGER.debug(f'{SessionID} | Creating saving directory for inputs')
    try:
        os.mkdir(output_dir)
    except FileExistsError:
        LOGGER.debug(f'{SessionID} | Directory already exists')
        pass

    ###################################
    # Calculating slope 1 (enhancement)
    LOGGER.debug(f'{SessionID} | Starting slope 1 calculation')
    Tmean = np.repeat(np.expand_dims(np.mean(T[:,:,:,0:2], axis=3), axis=-1), 2, axis=-1).astype(np.float32)
    Dmean = np.repeat(np.expand_dims(np.mean(D[:,:,:,0:2], axis=3), axis=-1), 2, axis=-1).astype(np.float32)
    slope1 = np.divide(
        np.sum((T[:,:,:,0:2] - Tmean) * (D[:,:,:,0:2] - Dmean), axis=3),
        np.sum(np.square((T[:,:,:,0:2] - Tmean)), axis=3)
    ).astype(np.float32)
    slope1 = slope1 / p95

    header['glmax'] = np.max(slope1)
    header['glmin'] = np.min(slope1)
    header['descrip'] = 'pre slp img'

    LOGGER.debug(f'{SessionID} | Slope 1 shape: {slope1.shape}')
    LOGGER.debug(f'{SessionID} | Header shape: {header.get_data_shape()}')

    nib.save(nib.Nifti1Image(slope1.astype('float32'), img.affine, header), output_dir + f'/slope1.nii.gz')
    LOGGER.debug(f'{SessionID} | Saved slope 1')

    ###################################
    # Calculating slope 2 (washout)
    LOGGER.debug(f'{SessionID} | Starting slope 2 calculation')
    Tmean = np.repeat(np.expand_dims(np.mean(T[:,:,:,1:], axis=3), axis=-1), len(Times)-1, axis=-1).astype(np.float32)
    Dmean = np.repeat(np.expand_dims(np.mean(D[:,:,:,1:], axis=3), axis=-1), len(Times)-1, axis=-1).astype(np.float32)
    slope2 = np.divide(
        np.sum((T[:,:,:,1:] - Tmean) * (D[:,:,:,1:] - Dmean), axis=3),
        np.sum(np.square((T[:,:,:,1:] - Tmean)), axis=3)
    ).astype(np.float32)
    slope2 = slope2 / p95

    header['glmax'] = np.max(slope2)
    header['glmin'] = np.min(slope2)
    header['descrip'] = 'post slp img'

    LOGGER.debug(f'{SessionID} | Slope 2 shape: {slope2.shape}')
    LOGGER.debug(f'{SessionID} | Header shape: {header.get_data_shape()}')

    nib.save(nib.Nifti1Image(slope2.astype('float32'), img.affine, header), output_dir + f'/slope2.nii.gz')
    LOGGER.debug(f'{SessionID} | Saved slope 2')

    ###################################
    # Creating post-contrast image
    LOGGER.debug(f'{SessionID} | Starting post contrast scan')
    img = nib.load(Fils[1])
    data1 = img.get_fdata().astype(np.float32)
    data1[np.isnan(data1)] = 0
    post = data1/p95

    LOGGER.debug(f'{SessionID} | Post contrast shape: {post.shape}')
    LOGGER.debug(f'{SessionID} | Header shape: {header.get_data_shape()}')

    nib.save(nib.Nifti1Image(post.astype('float32'), img.affine, img.header), output_dir + f'/post.nii.gz')
    LOGGER.debug(f'{SessionID} | Saved post contrast scan')

    ###################################

def generate_inputs_wrapper(SessionID): 
    try:
        generate_slopes(SessionID)
    except Exception as e:
        LOGGER.error(f'{SessionID} | Failed to generate inputs')
        LOGGER.error(f'{SessionID} | {e}')
        return


if __name__ == '__main__':
    try:
        Data_table = pd.read_csv('/scratch/joyliu/code/BreastCancerDiagnosisMRI/data/Data_table_timing.csv')
    except:
        LOGGER.error('MISSING CRITICAL FILE | "data_table_timing.csv"')
        exit()
     
    session = np.unique(Data_table['SessionID'])

    # Check if inputs have already been generated
    if os.path.exists(SAVE_DIR):
        print('Inputs already generated')
        # print('To reprocess data, please remove /data/inputs')
        # exit()
    else:
        # Create directory for saving inputs
        os.mkdir(SAVE_DIR)


    run_with_progress(generate_inputs_wrapper, session, Parallel=True)
