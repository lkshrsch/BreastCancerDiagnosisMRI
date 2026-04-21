#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Builds a timing table for breast MR series by scanning multiple root directories
for DICOM series, extracting per-series metadata, and writing a CSV compatible
with geninputs.py expectations.

Output: /FL_system/data/Data_table_timing.csv

Columns (at minimum):
- SessionID: stable identifier per study (PatientID_StudyDate_StudyTime)
- SeriesInstanceUID
- StudyInstanceUID
- PatientID
- Major: per-session order index (ascending AcquisitionDateTime/SeriesTime)
- TriTime: Trigger time in ms if available, else 'Unknown'
- ScanDur: Acquisition duration in us if available, else 'Unknown'
- AcqTime: acquisition time as HH:MM:SS
- Series_desc: SeriesDescription
- Modality
- DicomPath: directory containing this series (first detected instance)

This utility is conservative and resilient to mixed directory structures. It
parses only one instance per series to minimize IO, and assigns ordering after
gathering all series in a session.
"""

import os
from re import L
import sys
import traceback
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


try:
    import pydicom as pyd
    from pydicom.errors import InvalidDicomError
except Exception as exc:  # pragma: no cover
    print('ERROR: pydicom is required to run this script.\n', exc)
    sys.exit(1)


YAIB_BASE = '/mnt/shareddata/datasets/YAIB-cohorts/AIR_API_Downloads'
YAIB_SUBFOLDERS = [
    'missing_breast_mri_feb9_24',
    'chest_ct_breast_mr_missing',
]
AIR_EXTRACTED_ROOT = '/mnt/shareddata/datasets/AIR/extracted/'
AIR_DIRS_CSV = '/mnt/shareddata/datasets/breast_ucsf_mri/contrast_pixel_space/metadata/full_studies_dirs.csv'

OUTPUT_DIR = 'data'
OUTPUT_PATH = f'{OUTPUT_DIR}/Data_table_timing_complete.csv'


def is_probably_dicom(file_path: str) -> bool:
    """Lightweight DICOM detection to avoid expensive reads."""
    try:
        
        with open(file_path, 'rb') as f:
            prefix = f.read(132)
            return len(prefix) >= 132 and prefix[128:132] == b'DICM'
    except Exception:
        return False


def try_read_header(file_path: str):
    """Read a DICOM header without pixel data. Returns dataset or None."""
    try:
        return pyd.dcmread(file_path, stop_before_pixels=True, force=True)
    except (InvalidDicomError, Exception):
        return None


def hhmmss_to_hh_mm_ss(hhmmss: str) -> Optional[str]:
    """Convert DICOM time 'HHMMSS(.ffffff)' to 'HH:MM:SS'."""
    if not hhmmss:
        return None
    try:
        base = hhmmss.split('.')[0]
        base = base.zfill(6)
        return f"{base[0:2]}:{base[2:4]}:{base[4:6]}"
    except Exception:
        return None


def extract_series_record(file_path: str, ds, accession: Optional[str]) -> Optional[Dict]:
    """Extract per-series metadata from a single instance."""
    try:
        modality = getattr(ds, 'Modality', None)
        if modality and str(modality).upper() != 'MR':
            # print(f"Skipping {file_path} because it is {modality} series")
            return None

        series_uid = getattr(ds, 'SeriesInstanceUID', None)
        study_uid = getattr(ds, 'StudyInstanceUID', None)
        patient_id = getattr(ds, 'PatientID', None)
        series_num = getattr(ds, 'SeriesNumber', None)
        if accession is None:
            accession = getattr(ds, 'AccessionNumber', None)

        acq_date = getattr(ds, 'AcquisitionDate', None) or getattr(ds, 'StudyDate', None)
        acq_time = getattr(ds, 'AcquisitionTime', None) or getattr(ds, 'SeriesTime', None) or getattr(ds, 'ContentTime', None)
        acq_time_hms = hhmmss_to_hh_mm_ss(str(acq_time)) if acq_time else None

        tri_time = getattr(ds, 'TriggerTime', None)
        if tri_time is None:
            tri_time_val = 'Unknown'
        else:
            try:
                tri_time_val = float(tri_time)
            except Exception:
                tri_time_val = 'Unknown'

        acq_duration = getattr(ds, 'AcquisitionDuration', None)
        if acq_duration is None:
            scan_dur = 'Unknown'
        else:
            try:
                # Convert seconds -> microseconds to match expectations in geninputs.py path
                scan_dur = int(float(acq_duration) * 1e6)
            except Exception:
                scan_dur = 'Unknown'

        # SessionID based on PatientID + StudyDate + StudyTime if available, else StudyInstanceUID
        study_time = getattr(ds, 'StudyTime', None)
        study_time_hms = hhmmss_to_hh_mm_ss(str(study_time)) if study_time else None
        if patient_id and acq_date and study_time_hms:
            session_id = f"{str(patient_id)}_{str(acq_date)}_{study_time_hms.replace(':','')}"
        elif study_uid:
            session_id = str(study_uid)
        else:
            # Fallback: bucket by top directory name
            session_id = os.path.basename(os.path.dirname(file_path))

        record = {
            'SessionID': session_id,
            'SeriesInstanceUID': str(series_uid) if series_uid is not None else None,
            'StudyInstanceUID': str(study_uid) if study_uid is not None else None,
            'PatientID': str(patient_id) if patient_id is not None else None,
            'Orig Series #': int(series_num) if isinstance(series_num, (int, float)) else None,
            'sample_name': str(accession) if accession is not None else None,
            'TriTime': tri_time_val,
            'ScanDur': scan_dur,
            'AcqDate': str(acq_date) if acq_date is not None else None,
            'AcqTime': acq_time_hms if acq_time_hms is not None else 'Unknown',
            'Modality': str(modality) if modality is not None else None,
            'DicomPath': file_path,
        }
        return record
    except Exception as e:
        print(f"Error extracting series record from {file_path}: {e}")
        return None


def find_first_dicom_file(dir_path: str) -> Optional[str]:
    try:
        for fname in os.listdir(dir_path):
            fpath = os.path.join(dir_path, fname)
            if not os.path.isfile(fpath):
                continue
            if fname.lower().endswith('.dcm') or fname.upper().startswith('IM') or fname.upper().startswith('MR') or is_probably_dicom(fpath):
                return fpath
    except Exception:
        return None
    return None


def collect_series_dirs_under(root: str, accession: Optional[str]) -> List[str]:
    series_dirs: List[str] = []
    path = f"{root}/{accession}"
    if not os.path.exists(path):
        return series_dirs
    filenames = os.listdir(path)
    # Consider a directory a series if it contains at least one dicom-like file
    series_dirs = [os.path.join(path, fn) for fn in filenames if fn.lower().endswith('.dcm')]
    return series_dirs


def get_yaib_series_dirs(accession: Optional[str]) -> List[str]:
    dirs: List[str] = []
    base = YAIB_BASE
    if not os.path.exists(base):
        return dirs
    for sub in YAIB_SUBFOLDERS:
        root = os.path.join(base, sub)
        accession_dir = os.path.join(root, accession)
        if not os.path.exists(accession_dir):
            continue
        for sdir in os.listdir(os.path.join(root, accession)):
            series = os.path.join(root, accession, sdir)
            if os.path.isdir(series):
                for file in os.listdir(series):
                    sample = find_first_dicom_file(os.path.join(series, file))
                    if sample != None:
                        dirs.append(sample)
    return dirs


def get_air_series_dirs(accession: Optional[str], directory: Optional[str]) -> List[str]:
    dirs: List[str] = []
    if not os.path.exists(AIR_EXTRACTED_ROOT):
        return dirs

    dirs.extend(collect_series_dirs_under(AIR_EXTRACTED_ROOT, accession=directory))
    return dirs


def scan_series_from_dirs(series_dirs: List[str], metadata: pd.DataFrame) -> List[Dict]:
    records: List[Dict] = []
    for acc in tqdm(series_dirs):
        # sample = find_first_dicom_file(sdir)
        # if not sample:
        #     continue
        sdirs = series_dirs[acc]
        seen_phases: set = set()
        metadata_sample = metadata[metadata['sample_name'] == acc]
        for sdir in tqdm(sdirs):
            ds = try_read_header(sdir)
            if ds is None:
                continue

            rec = extract_series_record(sdir, ds, accession=acc)
            if rec is None:
                continue
            records.append(rec)
            metadata_rec = metadata_sample[metadata_sample['Orig Series #'] == rec['Orig Series #']]
            if len(metadata_rec) > 0:
                seen_phases.add(metadata_rec['Series'].iloc[0])
            if seen_phases == set(metadata_sample['Series']):
                break
            
    return records


def assign_major(records: List[Dict]) -> List[Dict]:
    """Assign per-session order based on AcquisitionDate+Time, then SeriesNumber."""
    by_session: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        if r.get('SessionID') is None:
            continue
        by_session[str(r['SessionID'])].append(r)

    for session_id, rows in by_session.items():
        def key_fn(r):
            date = r.get('AcqDate') or ''
            time = r.get('AcqTime') or '00:00:00'
            try:
                dt = datetime.strptime(f"{date} {time}", "%Y%m%d %H:%M:%S")
            except Exception:
                dt = datetime.min
            series_num = r.get('SeriesNumber') or 0
            return (dt, series_num)

        rows.sort(key=key_fn)
        for idx, r in enumerate(rows):
            r['Major'] = int(idx)

    # Flatten
    out = []
    for rows in by_session.values():
        out.extend(rows)
    return out


def write_csv(records: List[Dict], output_path: str, metadata: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame.from_records(records)
    # Reorder/select columns to match expectations and readability
    cols = [
        'SessionID', 'Major', 'PatientID', 'StudyInstanceUID', 'SeriesInstanceUID',
        'Orig Series #', 'sample_name', 'AcqDate', 'AcqTime', 'TriTime', 'ScanDur',
        'Series_desc', 'Modality', 'DicomPath',
    ]
    df = df.reindex(columns=cols)
    # # This works for the original files from Wynton (aka if they were in AIR)
    # metadata = pd.read_csv('/mnt/shareddata/datasets/breast_ucsf_mri/contrast_pixel_space/metadata/mri_series.csv')
    # metadata = metadata[metadata['Orig Study UID'] == df.iloc[0]['StudyInstanceUID']]
    # df_with_series_desc = df.merge(metadata, left_on='SeriesNumber', right_on='Orig Series #' )
    # if len(df_with_series_desc) == 0:
    #     print(f'No series description found for {df.iloc[0]["StudyInstanceUID"]}')
        
    #     metadata = pd.read_csv(f'/mnt/shareddata/datasets/breast_ucsf_mri/contrast_pixel_space/data/{df.iloc[0]["AccessionNumber"]}/converted.csv')
    #     first_row = list(metadata.columns)
    #     cols = ['Series', 'File', 'Orig Series #', 'Series Desc']
    #     metadata.columns = cols
    #     metadata = pd.concat([pd.DataFrame([first_row], columns=cols), metadata],ignore_index=True)
    #     df_with_series_desc = df.merge(metadata, left_on='SeriesNumber', right_on='Orig Series #' )
    df_with_series_desc = df.merge(metadata)
    df_with_series_desc.to_csv(output_path, index=False)
    
    print(f'Wrote timing table with {len(df_with_series_desc)} rows to: {output_path}')


def filter_records(records: List[Dict], accession: Optional[str], patient: Optional[str], session: Optional[str], path_substr: Optional[str]) -> List[Dict]:
    def keep(r: Dict) -> bool:
        if accession and str(r.get('AccessionNumber', '')) != str(accession):
            return False
        if patient and str(r.get('PatientID', '')) != str(patient):
            return False
        if session and str(r.get('SessionID', '')) != str(session):
            return False
        if path_substr and path_substr not in str(r.get('DicomPath', '')):
            return False
        return True
    return [r for r in records if keep(r)]


def get_series_desc(df: pd.DataFrame, filter_accession: Optional[str] = None) -> pd.DataFrame:
    series_descs = []
    # This works for the original files from Wynton (aka if they were in AIR)
    metadata = pd.read_csv('/mnt/shareddata/datasets/breast_ucsf_mri/contrast_pixel_space/metadata/mri_series.csv')
    pd.options.mode.copy_on_write = True 
    for _, row in tqdm(df.iterrows()):
        row_metadata = metadata[metadata['Orig Study UID'] == row['Orig Study UID']]
        row_metadata['sample_name'] = [row['sample_name']] * len(row_metadata)
        if len(row_metadata) == 0:
            try:
                row_path = f'/mnt/shareddata/datasets/breast_ucsf_mri/contrast_pixel_space/data/{row["sample_name"]}/converted.csv'
                if not os.path.exists(row_path):
                    print(f'{row["sample_name"]} | No metadata found')
                    continue
                row_metadata = pd.read_csv(row_path)
                first_row = list(row_metadata.columns)
                if len(first_row) == 4:
                    
                    cols = ['Series', 'File', 'Orig Series #', 'Orig Series Description']
                    
                elif len(first_row) == 3:
                    cols = ['Series', 'File', 'File Path']

                row_metadata.columns = cols
                row_metadata = pd.concat([pd.DataFrame([first_row], columns=cols), row_metadata],ignore_index=True)
                row_metadata['sample_name'] = [row['sample_name']] * len(row_metadata)
                if len(first_row) == 3:
                    series_num = []
                    for _, row_metadata in row_metadata.iterrows():
                        for subfolder in YAIB_SUBFOLDERS:
                            header = try_read_header(f"{YAIB_BASE}/{subfolder}/{row['sample_name']}/{row_metadata['File Path']}")
                            if header is not None:
                                series_num.append(header.SeriesNumber)
                                break
                            
                    row['Orig Series #'] = series_num
            except Exception as e:
                print(f'{row["sample_name"]} | {e}')
                continue
        series_descs.append(row_metadata)
    series = pd.concat(series_descs)
    series = series[series['Series'].isin(['T1FS', 'Ph1', 'Ph2', 'Ph5'])]

    return series
            
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Index DICOM MR series and build timing CSV')
    parser.add_argument('--accessions', default='/mnt/shareddata/users/joyliu/current/ReportAnalysis/converted_volumes/info/val_with_path_v5.csv')
    parser.add_argument('--output', default=OUTPUT_PATH, help='Output CSV path')
    parser.add_argument('--filter-accession', default=None, help='AccessionNumber to include')
    parser.add_argument('--filter-patient', default=None, help='PatientID to include')
    parser.add_argument('--filter-session', default=None, help='SessionID to include')
    args = parser.parse_args()

    directories = pd.read_csv(AIR_DIRS_CSV)
    accessions = pd.read_csv(args.accessions)
    directories['sample_name'] = directories['Orig Acc #'].astype(str)
    
    df = accessions.merge(directories, how='left')
    metadata = get_series_desc(df, filter_accession=args.filter_accession)
    
    acc_dicoms = {}
    # Stage 1: YAIB search (grouped by series)
    if args.filter_accession:
        series_dirs = get_yaib_series_dirs(args.filter_accession)
        if len(series_dirs) == 0:
            # Stage 2: AIR extracted using metadata CSV
            directory = directories[directories['sample_name'] == args.filter_accession].iloc[0]['Directory']
            print(directory)
            series_dirs = get_air_series_dirs(args.filter_accession, directory=directory)
        acc_dicoms[args.filter_accession] = series_dirs
        print(len(series_dirs))
        
        

    else:
        for _, row in df.iterrows():
            accession = row['sample_name']
            series_dirs = get_yaib_series_dirs(accession)
            if len(series_dirs) == 0:
                directory = row['Directory']
                # Stage 2: AIR extracted using metadata CSV
                series_dirs = get_air_series_dirs(accession, directory=directory)
            acc_dicoms[accession] = series_dirs
            
            if len(series_dirs) == 0:
                directory = directories[directories['sample_name'] == row['sample_name']]['Directory']
                if len(directory) == 0:
                    print(f'{accession} | No series dirs found')

                else: 
                    directory = directory.iloc[0]
                    series_dirs = get_air_series_dirs(accession, directory=directory)
                acc_dicoms[accession] = series_dirs
            
                
   
    records = scan_series_from_dirs(acc_dicoms, metadata)
    if not records:
        print('No MR series found in provided roots.')
        return
    # records = filter_records(records, args.filter_accession, args.filter_patient, args.filter_session, args.filter_path_substr)
    # if not records:
    #     print('No series matched filters.')
    #     return
    records = assign_major(records)
    write_csv(records, args.output, metadata)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted by user')
    except Exception as exc:
        print('Failed to build timing table:', exc)
        traceback.print_exc()

