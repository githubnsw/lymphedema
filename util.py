import os
import torch
import numpy as np
from PIL import Image
import SimpleITK as sitk
import nibabel as nib
import nrrd

# ============================================================
# Dataset Classes
# ============================================================
class Lym_Dataset(torch.utils.data.Dataset):
    CLASSES = ['background', 'muscle', 'fat']

    def __init__(self, img_path, mask_path, classes=None, transform=None, augmentation=None):
        self.ids = os.listdir(img_path)
        self.img_path = img_path
        self.mask_path = mask_path
        self.transform = transform
        self.augmentation = augmentation

        self.images_fps = [os.path.join(img_path, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(mask_path, image_id) for image_id in self.ids]
        # Convert class names to lowercase and assign indices (example)
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        filename = self.ids[idx]
        image_path = os.path.join(self.img_path, filename)
        mask_path = os.path.join(self.mask_path, filename)

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = self._preprocess_mask(mask)

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        sample = dict(image=image, mask=mask)
        if self.transform is not None:
            sample = self.transform(**sample)

        return sample

    @staticmethod
    def _preprocess_mask(mask):
        mask = mask.astype(np.float32)
        mask[mask == 2.0] = 2.0
        mask[(mask == 1.0) | (mask == 3.0)] = 1.0
        return mask


class SimpleLymDataset(Lym_Dataset):
    CLASSES = ['background', 'muscle', 'fat']

    def __getitem__(self, *args, **kwargs):
        sample = super().__getitem__(*args, **kwargs)
        # Resize image and mask to 512x512
        image = np.array(Image.fromarray(sample["image"]).resize((512, 512), Image.LINEAR))
        mask = np.array(Image.fromarray(sample["mask"]).resize((512, 512), Image.NEAREST))
        self.class_values = [0, 1, 2]
        # Convert from HWC to CHW format
        sample["image"] = np.moveaxis(image, -1, 0)
        # For the mask, create binary masks for each class and stack them
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        sample["mask"] = np.moveaxis(mask, -1, 0)
        return sample


# ============================================================
# Additional Utility Functions
# ============================================================

def compute_dice_coefficient(mask_gt, mask_pred):
    """
    Calculates the Dice coefficient between the ground truth and predicted masks.
    """
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    # Calculate the number of overlapping pixels using a logical AND (converting booleans to integers)
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2 * volume_intersect / volume_sum


def createFolder(directory):
    """
    Creates the specified directory if it does not exist.
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Unable to create directory", directory)


def volumetry(value, ps_0, ps_1):
    """
    Calculates the volume using the 2D pixel count (value) and the pixel spacing (ps_0, ps_1).
    (Here, we multiply by 0.01 for unit conversion.)
    """
    volume = value * float(ps_0) * float(ps_1) * 0.01
    return volume


def d3_volume(volume, sc):
    """
    Calculates the 3D volume using the 2D volume (volume) and the slice thickness (sc).
    (Here, we multiply by 0.1 for unit conversion.)
    """
    val = volume * sc * 0.1
    return val


def nrrd_to_nifti(nrrd_file, output_nifti_file):
    """
    Converts an NRRD file to NIfTI format and saves it.
    """
    image = sitk.ReadImage(nrrd_file)
    sitk.WriteImage(image, output_nifti_file)


def nifti_to_dicom(nifti_file):
    """
    Extracts the voxel spacing and slice thickness from a NIfTI file.
    It uses the header's zooms values, assuming the first two values represent in-plane spacing,
    and the third value represents the slice thickness.
    """
    nifti_img = nib.load(nifti_file)
    nifti_data = nifti_img.get_fdata()
    voxel_spacing = nifti_img.header.get_zooms()[:2]  # x, y spacing
    slice_thickness = nifti_img.header.get_zooms()[2]   # z spacing (slice thickness)
    return voxel_spacing, slice_thickness


def get_voxel_info_from_nrrd(nrrd_file):
    """
    Directly extracts the voxel spacing and slice thickness from an NRRD file.
    If the header contains 'spacings' information, it is used; otherwise, it is read using SimpleITK.
    """
    data, header = nrrd.read(nrrd_file)
    if 'spacings' in header:
        spacings = header['spacings']
        voxel_spacing = spacings[:2]
        slice_thickness = spacings[2] if len(spacings) >= 3 else 1.0
    else:
        image = sitk.ReadImage(nrrd_file)
        spacing = image.GetSpacing()  # (spacing_x, spacing_y, spacing_z)
        voxel_spacing = spacing[:2]
        slice_thickness = spacing[2]
    return voxel_spacing, slice_thickness
