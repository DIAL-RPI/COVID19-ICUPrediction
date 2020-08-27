# -*- coding: utf-8 -*-
"""
The script to demo the infection volume and infection ratio procedure

Objective: 
1. Read in the image from the resampled nii files (generated by the resize_volume.py)
                          the lung lobe segmentation files 
                          the lesion segmentation files
2. Segment the lesion regions within certain lung lobe
3. Calculate the infection ratio(IR) and infection volume(IV) of different regions and different HU ranges
4. Run the manuscript in a multi-processing way

Input:
1. Resampled CT images
2. Lung lobe masks ()
3. Lesion masks ()

Output:
IV and IR list of each lung lobe within four HU ranges of each patient/image
"""

from __future__ import print_function

import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor

import sys
import os
import os.path as osp
import shutil

from multiprocessing import Pool
from multiprocessing import TimeoutError as MP_TimeoutError
from time import sleep

import csv

START = "START"
FINISH = "FINISH"
WARNING = "WARNING"
ERROR = "ERROR"
LOG = "LOG"

# file locations
lobe_mask_loc = 'data_location' # lobe segmentation files
lesion_mask_loc = 'data_location' # lesion segmentation files
image_loc = 'data_location' # CT images
save_root_path = 'data_location' # file save location

# read nii file
def read_nii_file(file_loc):
    reader = sitk.ImageFileReader()
    reader.SetImageIO("NiftiImageIO")
    reader.SetFileName(file_loc)
    image = reader.Execute()
    return image

# read dicom files
def read_dcm_file(file_loc):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(file_loc)
    reader.SetFileNames(dicom_names)
    pcr_img = reader.Execute()
    return pcr_img

def log_print(pid, comment, logs):
    if type(logs) is str:
        logs = [logs]
    for log in logs:
        print("# JOB %d : --%s-- %s" % (pid, comment, log))
    sys.stdout.flush()


def result_scan(results):
    unfinish = 1
    while unfinish > 0:
        unfinish = 0
        for i, res in enumerate(results):
            try:
                res.get()
            except Exception as e:
                if type(e) == MP_TimeoutError:
                    unfinish += 1
                    continue
                else:
                    print("\n\n\nERROR OCCUR: PID ##%d##, ERRORTYPE: %s\n\n\n" %(i, type(e)))
                    raise e

def extract_IV_IR_features(input_file_loc, mask_lobe_loc, mask_lesion_loc, output_file_name, pid, title_flag = 1):
    # read input image
    pcr_img = read_nii_file(input_file_loc)
    pcr_img_np = sitk.GetArrayFromImage(pcr_img)
    # read lung lobe seg
    mask_lobe = read_nii_file(mask_lobe_loc)
    mask_lobe_np = sitk.GetArrayFromImage(mask_lobe)
    # read lesion seg
    mask_lesion = read_nii_file(mask_lesion_loc)
    
    mask_inter = mask_lobe * mask_lesion
    mask_inter_np = sitk.GetArrayFromImage(mask_inter)
    
    
    Lobe_num = 5 # number of lung lobe
    # calculate WIV, WIR, LIV, LIR,
    conditions = lambda x, y: {
                        x==0:np.where(y<=-750), \
                        x==1:np.where((y>-750) & (y<=-300)),\
                        x==2:np.where((y>-300) & (y<=50)), \
                        x==3:np.where(y>50)
                    }
        
    HU_winow = lambda x: {
                        x==0:'(-Inf, -750]', \
                        x==1:'(-750, -300]',\
                        x==2:'(-300, 50]', \
                        x==3: '(50, +Inf]'
                    }
    
    # Start to write the output file
    log_print(pid, START, 'Vol:%s' % input_file_loc)
    with open(output_file_name, 'w', newline='') as outfile:
        # loop for 4 HU windows
        for win_index in np.arange(4):
            # win_index = 0
            mask_win_np = np.zeros(shape=pcr_img_np.shape)
            index = conditions(win_index, pcr_img_np)[True]
            mask_win_np[index] = 1
            
            #mask_lobe_np_win = mask_lobe_np*mask_win_np
            mask_inter_np_win = mask_inter_np*mask_win_np
            
            RV = np.array(np.where( (mask_lobe_np>=1) & (mask_lobe_np<=3) )).size
            RIV = np.array(np.where( (mask_inter_np_win>=1) & (mask_inter_np_win<=3) )).size
            RIR = RIV/RV
        
            LV = np.array(np.where(mask_lobe_np>3)).size
            LIV = np.array(np.where(mask_inter_np_win>3)).size
            LIR = LIV/LV
            
            WIV = RIV+LIV
            WIR = WIV/(RV+LV)
            
            Lobe_IV = np.zeros(5)
            Lobe_IR = np.zeros(5)
            for ii in np.arange(Lobe_num)+1:
                # ii = 1
                lesion_ROI = np.array(np.where(mask_inter_np_win == ii)) # calculate the voxel number of each lesion region
                lung_ROI = np.array(np.where(mask_lobe_np == ii))
                lesion_ROI_size = lesion_ROI.size
                if  lesion_ROI_size >= 27: # at least 9 voxels
                    Lobe_IR[ii-1] = lesion_ROI_size/(lung_ROI.size)
                    Lobe_IV[ii-1] = lesion_ROI_size
                    
            
            keys, values = ['EnergyWindow','WIV','WIR','LIV','LIR', 'RIV','RIR',\
                            'Lobe#1_IV', 'Lobe#1_IR', 'Lobe#2_IV', 'Lobe#2_IR', \
                            'Lobe#3_IV', 'Lobe#3_IR', 'Lobe#4_IV', 'Lobe#4_IR', 'Lobe#5_IV', 'Lobe#5_IR'],\
                        [HU_winow(win_index)[True], WIV, WIR, LIV, LIR, RIV, RIR,\
                         Lobe_IV[0], Lobe_IR[0], Lobe_IV[1], Lobe_IR[1],\
                         Lobe_IV[2], Lobe_IR[2], Lobe_IV[3], Lobe_IR[3], Lobe_IV[4], Lobe_IR[4]]
                    
            if title_flag == 1:
                title_flag += 1
                csvwriter = csv.writer(outfile)
                csvwriter.writerow(keys)
                csvwriter.writerow(values)
            else:
                csvwriter = csv.writer(outfile)
                csvwriter.writerow(values)
    log_print(pid, FINISH, 'Vol:%s' % input_file_loc)



_ = None
while _ not in ['y', 'n']:
    _ = input('About to remove dir %s, START? [y/n]' % save_root_path).lower()

if _ == 'n':
    exit()
    

shutil.rmtree(save_root_path, ignore_errors=True)
os.makedirs(save_root_path)
processes = 8
pool = Pool(processes)
results = list()
pid = 0


datasetList = os.listdir(image_loc)
datasetList.sort()
for filename in datasetList:
    file = os.path.splitext(filename)[0]
    
    # image location
    input_file = os.path.join(image_loc, filename)
    # lesion mask location
    mask_lesion_file = os.path.join(lesion_mask_loc, filename)
    
    save_file = osp.join(save_root_path, file+'.csv')
    
    results.append(
        pool.apply_async(
            extract_IV_IR_features
            args=(input_file, mask_lobe_file, mask_lesion_file, save_file, pid)))
    pid += 1

pool.close()
result_scan(results)
pool.join()