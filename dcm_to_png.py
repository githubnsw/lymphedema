import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pydicom
import numpy as np

# -------------------------------
# Configuration
# -------------------------------
DICOM_PATH = 'DICOM_PATH'
SAVE_PATH = 'SAVE_PATH'
WINDOW_CENTER = 40
WINDOW_WIDTH = 400
IMAGE_SIZE = (512, 512)

# Ensure save path exists
os.makedirs(SAVE_PATH, exist_ok=True)


# -------------------------------
# DICOM to PNG Conversion
def convert_dicom_to_png(dicom_file, save_file):
        s=pydicom.dcmread(dicom_file)
        img2d = s.pixel_array
        hu_image = img2d * s['RescaleSlope'].value + s['RescaleIntercept'].value
        img_min = WINDOW_CENTER - WINDOW_WIDTH // 2
        img_max = WINDOW_CENTER + WINDOW_WIDTH // 2
        window_image = hu_image.copy()
        window_image[window_image < img_min] = img_min
        window_image[window_image > img_max] = img_max
        norm_image = np.array(window_image, dtype=np.float64)
        norm_image -= np.min(norm_image)
        norm_image /= np.max(norm_image)
        image_window_norm = np.expand_dims(norm_image, axis=2)
        image_ths = np.concatenate([image_window_norm, image_window_norm, image_window_norm], axis=2)  # (512, 512, 3)
        fi_image = image_ths * 255

        image = Image.fromarray(fi_image.astype('uint8'), 'RGB')
        image = image.resize((512, 512))
      #  print(roots.split('\\')[-1])
        image.save(save_file)


# -------------------------------
# Process All DICOM Files
# -------------------------------
if __name__ == "__main__":
    for root, _, files in os.walk(DICOM_PATH):
        for file in files:
            if file.endswith(".dcm"):
                dicom_file = os.path.join(root, file)
                print(dicom_file)
                save_file = file.split('.')[0]+'.png'
                print(save_file)
                convert_dicom_to_png(dicom_file, SAVE_PATH+save_file)
                print(f"Saved: {save_file}")
