import numpy as np
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import torch
import os
import cv2
#from sklearn.model_selection import train_test_split
#from torchvision.transforms import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader


class DatasetBuilder:
    def __init__(self, csv_path, img_dir, original_shape=(256, 1600)):
        self.csv_path = csv_path
        self.img_dir = Path(img_dir)
        self.original_shape = original_shape

    def build_df(self):
        """
        Build a dataset from img dir for all Images:
        Args:
            img_dir: this is the path of image directory
            csv_path: this is the path of csv file( that made by run length encoding)
        return:
            dataframe: for all images
        """
        df_csv = pd.read_csv(self.csv_path)
        df_csv["HasDefect"] = df_csv["EncodedPixels"].notna()

        images_with_defect = set(df_csv['ImageId'].unique())  # use set to get unique
        print("The number of image had defect from csv:", len(images_with_defect))
        # Get all images from the folder of images
        all_images = set([image for image in os.listdir(self.img_dir) if image.endswith((".jpg", ".png", ".jpeg"))])
        print(f"The number of images from Image training: {len(all_images)}")
        images_without_defect = all_images - images_with_defect
        print(f"The number of images without defect: {len(all_images) - len(images_with_defect)}")

        # Build a dataframe for all images
        rows = []
        for _, row in df_csv.iterrows():
            rows.append(row.to_dict())
        print("Len of First update:", len(rows))

        for img_id in images_without_defect:
            for class_id in range(1, 5):
                rows.append({
                    "ImageId": img_id,
                    "ClassId": class_id,
                    "EncodedPixels": '',  # Don't have defect = empty
                    "HasDefect": False
                })

        df_complete = pd.DataFrame(rows)

        print(f"\n Class distribution:")
        for cls in range(1, 5):
            cls_data = df_complete[df_complete['ClassId'] == cls]
            defect_count = cls_data['HasDefect'].sum()
            total_count = len(cls_data)
            pct = (defect_count / total_count) * 100
            print(f"   Class {cls}: {int(defect_count):5} defects / {total_count:5} total ({pct:5.1f}%)")

        return df_complete


class RLEprocessor:
    """RLE encoding and decoding"""

    @staticmethod
    def rle_decoder(rle_str, shape=(256, 1600)):
        rle_list = rle_str.split()  # convert to list
        assert isinstance(rle_list, list)  # if is list, be passed
        start, length = [np.asarray(x, dtype=int) for x in (rle_list[0::2], rle_list[1::2])]
        # start = [rle_list[x] - 1 for x in range(0, len(rle_list), 2)]
        # length = [rle_list[x] for x in range(1, len(rle_list), 2)]
        start -= 1
        ends = start + length
        image = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        for start, end in zip(start, ends):
            image[start:end] = 1
        return image.reshape(shape, order='F')

    @staticmethod
    def build_mask(df, image_id):
        mask = np.zeros((256, 1600, 4), dtype=np.uint8)

        for i in range(1, 5):
            rle = df.loc[
                (df["ImageId"] == image_id) & (df['ClassId'] == i), "EncodedPixels"
            ]
            if len(rle) > 0:
                rle = rle.values[0]
                if pd.notna(rle):
                    mask[:, :, i - 1] = RLEprocessor.rle_decoder(rle)
        return mask

    @staticmethod
    def rle_encoder(mask):
        pixels = mask.T.flatten(order='F')
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1  # Use 0 to covert tuple to nparray
        runs[1::2] -= runs[::2]
        return ' '.join(str(x) for x in runs)


class DataPreprocessor:
    """Load data, crop, and prepare data"""

    def __init__(self, csv_path, image_dir, output_dir, patch_size=256, stride=400):
        self.csv_path = csv_path
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.patch_size = patch_size
        self.stride = stride
        self.original_shape = (256, 1600)
        self.patch_shape = (256, 400)

    def load_csv(self):
        """"Load csv file and parse"""
        df = pd.read_csv(self.csv_path)
        df["HasDefect"] = df["EncodedPixels"].notna()
        return df

    def create_patches(self, df, save_images=True, save_masks=True):
        new_rows = []
        unique_image = df["ImageId"].unique()

        image_dir = self.output_dir / 'images'
        mask_dir = self.output_dir / 'masks'

        if save_images:
            image_dir.mkdir(parents=True, exist_ok=True)
        if save_masks:
            mask_dir.mkdir(parents=True, exist_ok=True)

        for img_id in tqdm(unique_image, desc="Cropping Images"):
            img_path = self.image_dir / img_id
            if not img_path.exists():
                continue

            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            img_data = df[df["ImageId"] == img_id]
            full_mask = RLEprocessor.build_mask(df, img_id)

            for patch_idx in range(4):
                start_col = patch_idx * self.stride
                enc_col = start_col + self.patch_size

                path_img = image[:, start_col:enc_col].copy()
                patch_id = f"{img_id[:-4]}_patch{patch_idx}.jpg"

                if save_images:
                    patch_path = image_dir / patch_id
                    patch_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(patch_path), path_img)

                # Process masks for each class
                patch_mask = full_mask[:, start_col:enc_col, :].copy()

                # Save multi_channel mask
                if save_masks:
                    mask_file = mask_dir / f"{patch_id}_mask.npy"
                    np.save(str(mask_file), patch_mask)

                # Create CSV row
                new_rows.append({
                    'ImageId': f"{patch_id}",
                    'MaskId': f"{patch_id}_mask.npy",
                    'HasDefect': patch_mask.max() > 0,
                    'NumDefectClasses': (patch_mask.max(axis=(0, 1)) > 0).sum()
                })
        return pd.DataFrame(new_rows)


class SteverstalDataset(Dataset):
    """PyTorch Dataset for Severstal"""

    def __init__(self, df, image_dir, mask_dir, transform=None, num_classes=4):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ===== IMAGE =====
        img_path = self.image_dir / row['ImageId']
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = np.expand_dims(image, axis=-1)  # (H, W, 1)

        # ===== MASK =====
        mask_path = self.mask_dir / row['MaskId']
        mask = np.load(str(mask_path))  # (H, W, 4)

        # ===== AUGMENT =====
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            # print(mask.shape)
            if mask.ndim == 3 and mask.shape[0] != 4:
                mask = mask.permute(2, 0, 1)
        else:
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask).permute(2, 0, 1).float()

        return {
            'image': image,  # (1, H, W)
            'mask': mask,  # (4, H, W)
            'image_id': row['ImageId']
        }


train_transform = A.Compose([
    # Focus defect
    A.CropNonEmptyMaskIfExists(256, 256, p=0.7),

    # Geometric
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),

    # Image quality
    A.CLAHE(p=0.3),
    A.GaussianBlur(p=0.2),

    # Resize cho model
    A.Resize(224, 224),

    # Normalize grayscale
    A.Normalize(mean=(0.5,), std=(0.5,)),

    # Convert to tensor
    ToTensorV2()
])
val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2()
])
