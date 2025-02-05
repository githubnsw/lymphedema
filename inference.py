#!/usr/bin/env python
"""
Reference code for volumetric segmentation evaluation.

Note:
- Model and dataset classes are imported from external modules (e.g., lym_model.py, util.py).
- Utility functions such as compute_dice_coefficient, volumetry, d3_volume,
  nrrd_to_nifti, and nifti_to_dicom are assumed to be defined in util.py.
"""

import os
import torch
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import nrrd

# Import external modules (the implementations should be provided separately)
from lym_model import LymModel  # Segmentation model (e.g., implemented using PyTorch Lightning)
from util import (
    compute_dice_coefficient,
    volumetry,
    d3_volume,
    nrrd_to_nifti,
    nifti_to_dicom
)

# -------------------------------
# Configuration
# -------------------------------
# Paths (adjust as needed for your environment)
DCM_PATH    = "DCM_PATH"
EXCEL_PATH  = "EXCEL_PATH+FILE_NAME"
MODEL_DIR   = "MODEL_DIR"
OUTPUT_NIFTI= "./output.nii.gz"

# Set device (GPU if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Model Loading
# -------------------------------
model = LymModel("unetplusplus", "efficientnet-b7", in_channels=3, out_classes=3)
model.to(device)
model_path = os.path.join(MODEL_DIR, 'best_model.pt')
model.load_state_dict(torch.load(model_path))
model.eval()  # Set model to evaluation mode

# -------------------------------
# Patient Identification & Excel Label Loading
# -------------------------------
# Load the Excel file containing CT slice range information
r_df = pd.read_excel(EXCEL_PATH, engine="openpyxl")

# Read all files in the DCM_PATH folder and extract patient IDs from files with .nrrd extension
files = os.listdir(DCM_PATH)
patient_list = []
for f in files:
    name_parts = f.split('.')
    if name_parts[-1] == 'nrrd':
        patient_list.append(name_parts[0])

# -------------------------------
# Main Processing Loop per Patient
# -------------------------------
# Process CT and segmentation data for each patient
for p in set(patient_list):
    # Set file paths for the CT image and segmentation mask
    dcm_file  = os.path.join(DCM_PATH, p + '.nrrd')
    seg_file  = os.path.join(DCM_PATH, p + '.seg.nrrd')

    # Read the NRRD files: CT volume and segmentation volume
    dcm, d_header = nrrd.read(dcm_file)
    masks, m_header = nrrd.read(seg_file)
    total_num = int(dcm.shape[2])

    # Convert NRRD to NIfTI and then extract voxel spacing and slice thickness
    nrrd_to_nifti(dcm_file, OUTPUT_NIFTI)
    voxel_spacing, slice_thickness = nifti_to_dicom(OUTPUT_NIFTI)
    print(f"Patient {p} - Voxel Spacing: {voxel_spacing}, Slice Thickness: {slice_thickness}")

    # Extract the slice range information for the patient from the Excel file
    for _, row in r_df.iterrows():
        if p == row['AnonymizedID']:
            ischial = total_num - int(row['Ischial tuberosity'])  # e.g., Ischial tuberosity, iliac crest
            talus   = total_num - int(row['Superior Talus'])
            iter_num = ischial - talus + 1
            break

    print(f"Processing patient {p}, slices from {talus} to {ischial}")

    # Initialize variables for Dice scores, volumes, and accumulators
    fibro_score_total = 0
    fat_score_total   = 0
    muscle_score_total = 0
    total_score_total = 0
    fat_volume = 0
    muscle_volume = 0
    total_volume = 0
    fibro_volume = 0
    fat_volume_sum    = 0
    muscle_volume_sum = 0
    fibro_volume_sum  = 0

    gt_fat_volume_sum    = 0
    gt_muscle_volume_sum = 0
    gt_fibro_volume_sum  = 0

    cnt = 0

    # -------------------------------
    # Process Each Slice within the Specified Range
    # -------------------------------
    for i in range(iter_num):
        slice_idx = i + talus

        # Extract the CT slice and the corresponding segmentation slice
        img = np.array(dcm[:, :, slice_idx])
        mg  = masks[:, :, slice_idx]

        # Create copies for later post-processing
        dcm_img     = img.copy()
        fibro_gt    = img.copy()
        fibro_pred  = img.copy()

        fat_gt      = mg.copy()
        muscle_gt   = mg.copy()
        background_gt = mg.copy()
        mask = mg.copy()

        # Optional: Prepare a visualization mask (for example purposes)
        mg_mask = mask * 50
        mg_mask = np.expand_dims(mg_mask, axis=2)
        mask_ths = np.concatenate([mg_mask, mg_mask, mg_mask], axis=2)  # (512, 512, 3)

        # Ground Truth preprocessing:
        # - For muscle: convert original label 1 to 2
        # - For fat: convert original label 2 to 1
        # - For background: label 3 is set to 0
        muscle_gt[muscle_gt != 1] = 0
        muscle_gt[muscle_gt == 1] = 2

        fat_gt[fat_gt != 2] = 0
        fat_gt[fat_gt == 2] = 1

        background_gt[background_gt != 3] = 0
        background_gt[background_gt == 3] = 0

        gt_mask = fat_gt + muscle_gt + background_gt

        # CT image preprocessing: HU clipping and normalization
        dcm_img[dcm_img < -360] = -360
        dcm_img[dcm_img > 440] = 440
        norm_image = dcm_img.astype(np.float64)
        norm_image -= np.min(norm_image)
        norm_image /= np.max(norm_image)

        # Create a 3-channel image for model input
        image_window_norm = np.expand_dims(norm_image, axis=2)
        image_ths = np.concatenate([image_window_norm, image_window_norm, image_window_norm], axis=2)
        fi_image = image_ths * 255

        # Convert the image to a PIL image then to grayscale using OpenCV
        pil_img = Image.fromarray(fi_image.astype('uint8'), 'RGB')
        numpy_image = np.array(pil_img)
        opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2GRAY)

        # -------------------------------
        # Model Inference
        # -------------------------------
        data = torch.from_numpy(opencv_image).to(torch.device('cuda'))
        model.eval()
        logits = model(data)
        pr_masks = logits.sigmoid()
        pr_mask = pr_masks.to(torch.device('cpu'))
        nnn = pr_mask.detach().numpy() * 2

        reduction_mask = np.squeeze(nnn, axis=0)
        mask_img = reduction_mask.transpose(1, 2, 0).astype(np.uint8)

        # Post-processing: assign labels using simple channel thresholding
        fat   = mask_img.copy()
        muscle = mask_img.copy()
        back  = mask_img.copy()

        # For fat region: if channel 1 is present, assign 1
        fat[mask_img[:, :, 0] > 0] = 0
        fat[mask_img[:, :, 2] > 0] = 0
        fat[mask_img[:, :, 1] > 0] = 1

        # For muscle region: if channel 2 is present, assign 2
        muscle[mask_img[:, :, 0] > 0] = 0
        muscle[mask_img[:, :, 1] > 0] = 0
        muscle[mask_img[:, :, 2] > 0] = 2

        # For background (0) processing
        back[mask_img[:, :, 2] > 0] = 0
        back[mask_img[:, :, 1] > 0] = 0
        back[mask_img[:, :, 0] > 0] = 0

        pred = fat + back + muscle

        # Convert the masks to grayscale for Dice calculation
        fat_img    = cv2.cvtColor(fat, cv2.COLOR_RGB2GRAY)
        muscle_img = cv2.cvtColor(muscle, cv2.COLOR_RGB2GRAY)
        pred_img   = cv2.cvtColor(pred, cv2.COLOR_RGB2GRAY)

        # -------------------------------
        # Fibrosis Preprocessing
        # -------------------------------
        # For fibrotic region: set non-fat areas to -1000 and clip HU values accordingly
        fibro_pred[fat_img == 0] = -1000
        fibro_pred[fibro_pred < 0] = -1000
        fibro_pred[fibro_pred > 50] = -1000

        fibro_gt[fat_gt == 0] = -1000
        fibro_gt[fibro_gt < 0] = -1000
        fibro_gt[fibro_gt > 50] = -1000

        # -------------------------------
        # Dice Coefficient Calculation
        # -------------------------------
        score = compute_dice_coefficient(fibro_gt, fibro_pred)
        fibro_score_total += score

        score = compute_dice_coefficient(fat_gt, fat_img)
        fat_score_total += score

        score = compute_dice_coefficient(muscle_gt, muscle_img)
        muscle_score_total += score

        score = compute_dice_coefficient(gt_mask, pred_img)
        total_score_total += score

        # -------------------------------
        # Volume Calculation
        # -------------------------------
        p_fat_c = np.where(fat_img > 0)
        p_fat_v = volumetry(len(p_fat_c[1]), voxel_spacing[0], voxel_spacing[1])
        p_muscle_c = np.where(muscle_img > 0)
        p_muscle_v = volumetry(len(p_muscle_c[1]), voxel_spacing[0], voxel_spacing[1])
        p_fibro_c = np.where(fibro_pred > 0)
        p_fibro_v = volumetry(len(p_fibro_c[1]), voxel_spacing[0], voxel_spacing[1])

        fat_volume     += p_fat_v
        muscle_volume  += p_muscle_v
        fibro_volume   += p_fibro_v

        g_fat_c = np.where(fat_gt > 0)
        g_fat_v = volumetry(len(g_fat_c[1]), voxel_spacing[0], voxel_spacing[1])
        g_muscle_c = np.where(muscle_gt > 0)
        g_muscle_v = volumetry(len(g_muscle_c[1]), voxel_spacing[0], voxel_spacing[1])
        g_fibro_c = np.where(fibro_gt > 0)
        g_fibro_v = volumetry(len(g_fibro_c[1]), voxel_spacing[0], voxel_spacing[1])

        # Accumulate the ground truth volumes
        gt_fat_volume_sum    += g_fat_v
        gt_muscle_volume_sum += g_muscle_v
        gt_fibro_volume_sum  += g_fibro_v

        cnt += 1

    # -------------------------------
    # Average Dice and 3D Volume Calculation (over all slices)
    # -------------------------------
    avg_fat_dice    = fat_score_total / cnt
    avg_muscle_dice = muscle_score_total / cnt
    avg_fibro_dice  = fibro_score_total / cnt
    avg_total_dice  = total_score_total / cnt

    pred_muscle_volume_3d = d3_volume(muscle_volume, slice_thickness)
    pred_fat_volume_3d    = d3_volume(fat_volume, slice_thickness)
    pred_fibro_volume_3d  = d3_volume(fibro_volume, slice_thickness)
    pred_total_volume_3d  = pred_muscle_volume_3d + pred_fat_volume_3d

    gt_muscle_volume_3d = d3_volume(gt_muscle_volume_sum, slice_thickness)
    gt_fat_volume_3d    = d3_volume(gt_fat_volume_sum, slice_thickness)
    gt_fibro_volume_3d  = d3_volume(gt_fibro_volume_sum, slice_thickness)
    gt_total_volume_3d  = gt_muscle_volume_3d + gt_fat_volume_3d

    # -------------------------------
    # Print Results
    # -------------------------------
    print(f"\nPatient {p} results:")
    print("Predicted Volumes:")
    print(f"  Muscle Volume: {pred_muscle_volume_3d}")
    print(f"  Fat Volume:    {pred_fat_volume_3d}")
    print(f"  Fibro Volume:  {pred_fibro_volume_3d}")
    print(f"  Total Volume:  {pred_total_volume_3d}")
    print("Ground Truth Volumes:")
    print(f"  Muscle Volume: {gt_muscle_volume_3d}")
    print(f"  Fat Volume:    {gt_fat_volume_3d}")
    print(f"  Fibro Volume:  {gt_fibro_volume_3d}")
    print(f"  Total Volume:  {gt_total_volume_3d}")
    print("Average Dice Scores:")
    print(f"  Fat Dice:     {avg_fat_dice}")
    print(f"  Muscle Dice:  {avg_muscle_dice}")
    print(f"  Fibrosis Dice:{avg_fibro_dice}")
    print(f"  Total Dice:   {avg_total_dice}")
    print(f"Total slices processed: {cnt}\n")
