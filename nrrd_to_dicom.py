import os
import cv2
import numpy as np
import shutil
import pandas as pd
import SimpleITK as sitk
import nrrd

# -------------------------------
# Configuration
# -------------------------------
NRRD_PATH = 'NRRD_PATH'
OUTPUT_DICOM_DIR = 'OUTPUT_DICOM_DIR_PATH'
OUTPUT_PNG_DIR = 'OUTPUT_PNG_DIR_PATH'
CT_SLICE_RANGE_FILE  = 'CT_SLICE_RANGE_FILE_PATH'

# Ensure output directories exist
os.makedirs(OUTPUT_DICOM_DIR, exist_ok=True)
os.makedirs(OUTPUT_PNG_DIR, exist_ok=True)


# -------------------------------
# Utility Functions
# -------------------------------
def nrrd_to_dicom(nrrd_slice, output_dicom_path):
    """ Converts a single NRRD slice to DICOM format. """
    image = sitk.GetImageFromArray(nrrd_slice)
    sitk.WriteImage(image, output_dicom_path)


def nrrd_to_png(nrrd_slice, output_png_path):
    """ Converts a single NRRD slice to PNG format. """
    slice_img = ((nrrd_slice - nrrd_slice.min()) / (nrrd_slice.max() - nrrd_slice.min()) * 255).astype(np.uint8)
    cv2.imwrite(output_png_path, slice_img)


# -------------------------------
# Processing NRRD Files
# -------------------------------
if __name__ == "__main__":
    files = os.listdir(NRRD_PATH)
    patient_list = []

    r_df = pd.read_excel(CT_SLICE_RANGE_FILE, engine="openpyxl")

    for f in files:
        name = f.split('.')
        if name[-1] == 'nrrd':
            patient_list.append(name[0])

    for p in set(patient_list):
        dcm, d_header = nrrd.read(os.path.join(NRRD_PATH, p + '.nrrd'))
        masks, m_header = nrrd.read(os.path.join(NRRD_PATH, p + '.seg.nrrd'))

        total_num = dcm.shape[2]

        for index, row in r_df.iterrows():
            if p == row['AnonymizedID']:
                ischial = total_num - int(row['Ischial tuberosity'])
                talus = total_num - int(row['Superior Talus'])
                break

        print(f"Processing patient {p}, slices range: {talus} to {ischial}")

        for i in range(talus, ischial + 1):
            dicom_output_path = os.path.join(OUTPUT_DICOM_DIR, f"{p}_{i:05d}.dcm")
            png_output_path = os.path.join(OUTPUT_PNG_DIR, f"{p}_{i:05d}.png")

            nrrd_to_dicom(dcm[:, :, i], dicom_output_path)
            nrrd_to_png(masks[:, :, i], png_output_path)
