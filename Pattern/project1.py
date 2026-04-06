"""
================================================================================
SEVERSTAL STEEL DEFECT DETECTION - COMPLETE PROJECT
================================================================================
Pipeline:
1. Data Preprocessing (RLE → NumPy Masks)
2. Image Cropping (256×1600 → 4× 256×400)
3. Dataset Preparation
4. U-Net Segmentation Model
5. Training with Loss Functions
6. Evaluation & Metrics
7. Inference & Visualization
8. Kaggle Submission Format
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score, dice_score, precision_score, recall_score
import json
from datetime import datetime


# ============================================================================
# 1. RLE PROCESSOR - Chuyển đổi RLE ↔ Mask
# ============================================================================

class RLEProcessor:
    """RLE encoding/decoding utilities"""

    @staticmethod
    def rle_decode(rle_str, shape=(256, 1600)):
        """Decode RLE string to binary mask"""
        if pd.isna(rle_str) or rle_str == '':
            return np.zeros(shape, dtype=np.uint8)

        rle_list = rle_str.split()
        start, length = [np.asarray(x, dtype=int) for x in (rle_list[0::2], rle_list[1::2])]
        start -= 1
        ends = start + length

        image = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        for lo, hi in zip(start, ends):
            image[lo:hi] = 1

        return image.reshape(shape, order='F')

    @staticmethod
    def rle_encode(mask):
        """Encode binary mask to RLE string - dùng cho Kaggle submission"""
        pixels = mask.T.flatten(order='F')
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]

        return ' '.join(str(x) for x in runs)

    @staticmethod
    def build_mask(df, image_id):
        """Build 4-channel mask (H, W, 4) from RLE data"""
        mask = np.zeros((256, 1600, 4), dtype=np.uint8)

        for class_id in range(1, 5):
            rle = df.loc[(df["ImageId"] == image_id) & (df['ClassId'] == class_id), "EncodedPixels"]

            if len(rle) > 0:
                rle_value = rle.values[0]
                if pd.notna(rle_value):
                    mask[:, :, class_id - 1] = RLEProcessor.rle_decode(rle_value)

        return mask


# ============================================================================
# 2. DATA PREPROCESSOR - Load, Crop, Prepare Data
# ============================================================================

class DataPreprocessor:
    """Load, crop images, and prepare data"""

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
        """Load CSV and parse"""
        print("\n" + "=" * 70)
        print("📂 LOADING DATA")
        print("=" * 70)

        df = pd.read_csv(self.csv_path)
        df[['ImageId', 'ClassId']] = df['ImageId_ClassId'].str.rsplit('_', 1, expand=True)
        df['ClassId'] = df['ClassId'].astype(int)
        df['HasDefect'] = df['EncodedPixels'].notna()

        print(f"✅ Loaded {len(df)} records")
        print(f"✅ Unique images: {df['ImageId'].nunique()}")
        print(f"✅ Classes: {sorted(df['ClassId'].unique())}")

        return df

    def analyze_data(self, df):
        """Analyze dataset statistics"""
        print("\n" + "=" * 70)
        print("📊 DATA ANALYSIS")
        print("=" * 70)

        # Class distribution
        class_counts = df.groupby('ClassId')['HasDefect'].sum()
        class_names = {1: 'Crazing', 2: 'Inclusion', 3: 'Patches', 4: 'Scratches'}

        print("\n📈 Defects per class:")
        total_defects = 0
        for cls in sorted(df['ClassId'].unique()):
            count = class_counts[cls]
            name = class_names.get(cls, f"Class {cls}")
            pct = (count / len(df[df['ClassId'] == cls])) * 100
            print(f"   Class {cls} ({name:12}): {int(count):5} defects ({pct:5.1f}%)")
            total_defects += count

        # Images statistics
        imgs_with_defect = df[df['HasDefect']].groupby('ImageId').size()
        imgs_without_defect = df['ImageId'].nunique() - len(imgs_with_defect)

        print(f"\n📸 Image statistics:")
        print(f"   Total images: {df['ImageId'].nunique()}")
        print(f"   With defect: {len(imgs_with_defect)} ({len(imgs_with_defect) / df['ImageId'].nunique() * 100:.1f}%)")
        print(f"   Without defect: {imgs_without_defect} ({imgs_without_defect / df['ImageId'].nunique() * 100:.1f}%)")

        # Visualize
        self._plot_analysis(df, class_names, class_counts)

        return class_counts, class_names

    def _plot_analysis(self, df, class_names, class_counts):
        """Plot data analysis"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle('Severstal Dataset Analysis', fontsize=14, fontweight='bold')

        # Plot 1: Class distribution
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        axes[0].bar(range(1, 5), class_counts.values, color=colors, alpha=0.8, edgecolor='black')
        axes[0].set_xlabel('Defect Class')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Defects per Class')
        axes[0].set_xticks(range(1, 5))
        for i, v in enumerate(class_counts.values):
            axes[0].text(i + 1, v, str(int(v)), ha='center', va='bottom', fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)

        # Plot 2: Pie chart
        wedges, texts, autotexts = axes[1].pie(
            class_counts.values,
            labels=[f"Class {i}\n({class_names.get(i, f'Class {i}')})" for i in range(1, 5)],
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )
        axes[1].set_title('Class Distribution')
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        # Plot 3: Images with/without defects
        imgs_with = df[df['HasDefect']].groupby('ImageId').size().shape[0]
        imgs_without = df['ImageId'].nunique() - imgs_with
        axes[2].bar(['With Defect', 'Without Defect'], [imgs_with, imgs_without],
                    color=['#FF6B6B', '#90EE90'], alpha=0.8, edgecolor='black')
        axes[2].set_ylabel('Count')
        axes[2].set_title('Images with/without Defects')
        axes[2].grid(axis='y', alpha=0.3)
        for i, v in enumerate([imgs_with, imgs_without]):
            axes[2].text(i, v, str(v), ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig('./outputs/01_data_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n✅ Analysis plot saved to './outputs/01_data_analysis.png'")
        plt.close()

    def create_patches(self, df, save_images=True, save_masks=True):
        """Convert 256×1600 images into 4× 256×400 patches with multi-channel masks"""
        print("\n" + "=" * 70)
        print("✂️  CREATING IMAGE PATCHES")
        print("=" * 70)

        new_rows = []
        unique_images = df['ImageId'].unique()

        image_dir = self.output_dir / 'images'
        mask_dir = self.output_dir / 'masks'

        if save_images:
            image_dir.mkdir(parents=True, exist_ok=True)
        if save_masks:
            mask_dir.mkdir(parents=True, exist_ok=True)

        for img_id in tqdm(unique_images, desc="Processing images"):
            img_path = self.image_dir / img_id
            if not img_path.exists():
                continue

            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            # Build full 4-channel mask once
            full_mask = RLEProcessor.build_mask(df, img_id)

            # Create 4 patches
            for patch_idx in range(4):
                start_col = patch_idx * self.stride
                end_col = start_col + self.patch_size

                patch_img = image[:, start_col:end_col].copy()
                patch_id = f"{img_id[:-4]}_patch{patch_idx}"

                # Save image
                if save_images:
                    img_file = image_dir / f"{patch_id}.jpg"
                    cv2.imwrite(str(img_file), patch_img)

                # Crop multi-channel mask
                patch_mask = full_mask[:, start_col:end_col, :].copy()

                # Save multi-channel mask
                if save_masks:
                    mask_file = mask_dir / f"{patch_id}_mask.npy"
                    np.save(str(mask_file), patch_mask)

                # Create CSV row
                new_rows.append({
                    'ImageId': f"{patch_id}.jpg",
                    'MaskId': f"{patch_id}_mask.npy",
                    'HasDefect': patch_mask.max() > 0,
                    'NumDefectClasses': (patch_mask.max(axis=(0, 1)) > 0).sum()
                })

        df_patches = pd.DataFrame(new_rows)

        print(f"\n✅ Created {len(df_patches)} patch records")
        print(f"✅ Images saved to: {image_dir}")
        print(f"✅ Masks saved to: {mask_dir}")
        print(f"✅ Defect statistics:")
        print(f"   - Images with defects: {df_patches['HasDefect'].sum()}")
        print(f"   - Images without defects: {(~df_patches['HasDefect']).sum()}")

        return df_patches


# ============================================================================
# 3. PYTORCH DATASET
# ============================================================================

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

        # Load image
        img_path = self.image_dir / row['ImageId']
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Normalize image
        image = image.astype(np.float32) / 255.0

        # Load multi-channel mask
        mask_path = self.mask_dir / row['MaskId']
        masks = np.load(str(mask_path))  # (H, W, 4)

        # Convert to tensors
        image_tensor = torch.from_numpy(image).float().unsqueeze(0)  # (1, H, W)
        masks_tensor = torch.from_numpy(masks.transpose(2, 0, 1)).float()  # (4, H, W)

        return {
            'image': image_tensor,
            'masks': masks_tensor,
            'image_id': row['ImageId']
        }


# ============================================================================
# 4. U-NET SEGMENTATION MODEL
# ============================================================================

class DoubleConv(nn.Module):
    """Double convolution block"""

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """U-Net for semantic segmentation"""

    def __init__(self, in_channels=1, num_classes=4):
        super(UNet, self).__init__()

        # Encoder
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.middle = DoubleConv(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv_up4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv_up3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv_up2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_up1 = DoubleConv(128, 64)

        self.final_conv = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3))

        # Bottleneck
        middle = self.middle(self.pool4(d4))

        # Decoder
        u4 = self.up4(middle)
        u4 = torch.cat([u4, d4], dim=1)
        u4 = self.conv_up4(u4)

        u3 = self.up3(u4)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.conv_up3(u3)

        u2 = self.up2(u3)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.conv_up2(u2)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.conv_up1(u1)

        return self.final_conv(u1)


# ============================================================================
# 5. LOSS FUNCTIONS
# ============================================================================

class DiceLoss(nn.Module):
    """Dice Loss for semantic segmentation"""

    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)

        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (inputs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / \
               (inputs_flat.sum() + targets_flat.sum() + self.smooth)

        return 1 - dice


class BCEDiceLoss(nn.Module):
    """Combined BCE + Dice Loss"""

    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        dice_loss = self.dice_loss(inputs, targets)

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# ============================================================================
# 6. TRAINING ENGINE
# ============================================================================

class SegmentationTrainer:
    """Trainer for semantic segmentation"""

    def __init__(self, model, device, output_dir='./outputs'):
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_dice': [],
            'val_iou': []
        }

    def train_epoch(self, train_loader, criterion, optimizer, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}", leave=False)
        for batch in pbar:
            images = batch['image'].to(self.device)
            masks = batch['masks'].to(self.device)

            # Forward
            outputs = self.model(images)

            # Loss - calculate per class and average
            loss = 0
            for c in range(masks.shape[1]):
                loss += criterion(outputs[:, c:c + 1, :, :], masks[:, c:c + 1, :, :])
            loss = loss / masks.shape[1]

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item(): .4
            f})

            avg_loss = total_loss / len(train_loader)
            return avg_loss

        @torch.no_grad()
        def validate(self, val_loader, criterion):
            """Validate"""
            self.model.eval()
            total_loss = 0.0
            all_dices = []
            all_ious = []

            pbar = tqdm(val_loader, desc="Validating", leave=False)
            for batch in pbar:
                images = batch['image'].to(self.device)
                masks = batch['masks'].to(self.device)

                outputs = self.model(images)

                loss = 0
                for c in range(masks.shape[1]):
                    loss += criterion(outputs[:, c:c + 1, :, :], masks[:, c:c + 1, :, :])
                loss = loss / masks.shape[1]

                total_loss += loss.item()

                # Calculate metrics
                preds = torch.sigmoid(outputs) > 0.5

                for c in range(masks.shape[1]):
                    pred_c = preds[:, c].cpu().numpy().astype(np.uint8).flatten()
                    mask_c = masks[:, c].cpu().numpy().astype(np.uint8).flatten()

                    dice = self._dice_coefficient(pred_c, mask_c)
                    iou = jaccard_score(mask_c, pred_c, zero_division=0)

                    all_dices.append(dice)
                    all_ious.append(iou)

                pbar.update(1)

            avg_loss = total_loss / len(val_loader)
            avg_dice = np.mean(all_dices)
            avg_iou = np.mean(all_ious)

            return avg_loss, avg_dice, avg_iou

        @staticmethod
        def _dice_coefficient(pred, target):
            """Calculate Dice coefficient"""
            intersection = np.sum(pred * target)
            union = np.sum(pred) + np.sum(target)
            return (2 * intersection + 1e-6) / (union + 1e-6)

        def train(self, train_loader, val_loader, num_epochs=30, learning_rate=1e-3, patience=10):
            """Full training loop"""
            criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )

            best_val_loss = float('inf')
            best_model_path = self.output_dir / 'best_model.pt'
            patience_counter = 0

            print("\n" + "=" * 70)
            print("🚀 STARTING TRAINING")
            print("=" * 70)

            for epoch in range(num_epochs):
                print(f"\n📊 Epoch {epoch + 1}/{num_epochs}")

                # Train
                train_loss = self.train_epoch(train_loader, criterion, optimizer, epoch)

                # Validate
                val_loss, val_dice, val_iou = self.validate(val_loader, criterion)

                # Update history
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['val_dice'].append(val_dice)
                self.history['val_iou'].append(val_iou)

                print(f"   Train Loss: {train_loss:.4f}")
                print(f"   Val Loss: {val_loss:.4f}")
                print(f"   Val Dice: {val_dice:.4f}")
                print(f"   Val IoU: {val_iou:.4f}")

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(self.model.state_dict(), best_model_path)
                    print(f"   ✅ Best model saved!")
                else:
                    patience_counter += 1

                scheduler.step(val_loss)

                # Early stopping
                if patience_counter >= patience:
                    print(f"\n⚠️  Early stopping at epoch {epoch + 1}")
                    break

            print("\n✨ Training complete!")
            return best_model_path

        def plot_history(self, output_path='./outputs/02_training_history.png'):
            """Plot training history"""
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            fig.suptitle('Training History', fontsize=14, fontweight='bold')

            # Loss
            axes[0].plot(self.history['train_loss'], label='Train', linewidth=2)
            axes[0].plot(self.history['val_loss'], label='Val', linewidth=2)
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Loss')
            axes[0].legend()
            axes[0].grid(alpha=0.3)

            # Dice Score
            axes[1].plot(self.history['val_dice'], color='green', linewidth=2)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Dice Score')
            axes[1].set_title('Validation Dice Score')
            axes[1].grid(alpha=0.3)

            # IoU
            axes[2].plot(self.history['val_iou'], color='orange', linewidth=2)
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('IoU')
            axes[2].set_title('Validation IoU')
            axes[2].grid(alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✅ Training history saved to {output_path}")
            plt.close()

    # ============================================================================
    # 7. INFERENCE & PREDICTION
    # ============================================================================

    class SegmentationInference:
        """Inference module for segmentation"""

        def __init__(self, model, device):
            self.model = model
            self.device = device

        @torch.no_grad()
        def predict(self, image, threshold=0.5):
            """
            Predict segmentation masks

            Args:
                image (np.ndarray): Grayscale image (H, W)
                threshold (float): Threshold for binary mask

            Returns:
                dict: Predicted masks for each class
            """
            self.model.eval()

            # Preprocess
            image_normalized = image.astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_normalized).float().unsqueeze(0).unsqueeze(0)
            image_tensor = image_tensor.to(self.device)

            # Predict
            outputs = self.model(image_tensor)
            predictions = torch.sigmoid(outputs) > threshold

            # Post-process
            masks = {}
            for c in range(outputs.shape[1]):
                mask = predictions[0, c].cpu().numpy().astype(np.uint8)
                masks[f'class_{c + 1}'] = mask

            return masks

        def visualize_predictions(self, image, masks, output_path='./outputs/03_prediction_sample.png'):
            """Visualize predictions"""
            fig, axes = plt.subplots(1, 5, figsize=(16, 3))
            fig.suptitle('Segmentation Predictions', fontsize=12, fontweight='bold')

            axes[0].imshow(image, cmap='gray')
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            class_names = {
                'class_1': 'Crazing',
                'class_2': 'Inclusion',
                'class_3': 'Patches',
                'class_4': 'Scratches'
            }

            for idx, (class_name, mask) in enumerate(masks.items(), 1):
                display_name = class_names.get(class_name, class_name)
                axes[idx].imshow(mask, cmap='hot')
                axes[idx].set_title(f'{display_name}\n(Class {idx})')
                axes[idx].axis('off')

            plt.tight_layout()
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            print(f"✅ Visualization saved to {output_path}")
            plt.close()

    # ============================================================================
    # 8. KAGGLE SUBMISSION
    # ============================================================================

    class KaggleSubmissionBuilder:
        """Build Kaggle submission from predictions"""

        @staticmethod
        def masks_to_rle(predicted_masks_dir, output_csv='./outputs/submission.csv'):
            """
            Convert predicted masks to RLE format for Kaggle

            Args:
                predicted_masks_dir: Folder with predicted_masks/*.npy
                output_csv: Output CSV path
            """
            submission_rows = []
            mask_files = list(Path(predicted_masks_dir).glob('*.npy'))

            print(f"\n📝 Converting {len(mask_files)} masks to RLE format...")

            for mask_file in tqdm(mask_files):
                patch_id = mask_file.stem.replace('_mask', '')
                predicted_mask = np.load(mask_file)  # (H, W, 4)

                # Convert each channel to RLE
                for class_id in range(1, 5):
                    mask_channel = predicted_mask[:, :, class_id - 1]
                    rle = RLEProcessor.rle_encode(mask_channel)

                    submission_rows.append({
                        'ImageId_ClassId': f"{patch_id}_{class_id}",
                        'EncodedPixels': rle if rle else ''
                    })

            df_submission = pd.DataFrame(submission_rows)
            df_submission.to_csv(output_csv, index=False)
            print(f"✅ Submission CSV saved to {output_csv}")
            print(f"   Total rows: {len(df_submission)}")

            return df_submission

    # ============================================================================
    # 9. MAIN PIPELINE
    # ============================================================================

    def main():
        """Main pipeline"""

        # Configuration
        CONFIG = {
            'data_dir': './data',
            'train_csv': './data/train.csv',
            'train_images': './data/train_images',
            'output_dir': './outputs',
            'patch_dir': './data/patches',

            'batch_size': 8,
            'num_epochs': 30,
            'learning_rate': 1e-3,
            'num_classes': 4,
            'train_split': 0.8,

            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        }

        # Create output directory
        Path(CONFIG['output_dir']).mkdir(parents=True, exist_ok=True)
        Path(CONFIG['patch_dir']).mkdir(parents=True, exist_ok=True)

        print("\n" + "🚀 " * 25)
        print("SEVERSTAL STEEL DEFECT DETECTION - COMPLETE PIPELINE")
        print("🚀 " * 25)

        # ========== STEP 1: DATA PREPROCESSING ==========
        print("\n" + "=" * 70)
        print("STEP 1: DATA PREPROCESSING")
        print("=" * 70)

        preprocessor = DataPreprocessor(
            csv_path=CONFIG['train_csv'],
            image_dir=CONFIG['train_images'],
            output_dir=CONFIG['patch_dir']
        )

        df = preprocessor.load_csv()
        preprocessor.analyze_data(df)
        df_patches = preprocessor.create_patches(df, save_images=True, save_masks=True)
        df_patches.to_csv(os.path.join(CONFIG['patch_dir'], 'train_patches.csv'), index=False)

        # ========== STEP 2: DATA SPLIT ==========
        print("\n" + "=" * 70)
        print("STEP 2: DATA SPLIT & DATALOADERS")
        print("=" * 70)

        train_df, val_df = train_test_split(
            df_patches, test_size=1 - CONFIG['train_split'], random_state=42
        )

        train_dataset = SteverstalDataset(
            train_df,
            image_dir=os.path.join(CONFIG['patch_dir'], 'images'),
            mask_dir=os.path.join(CONFIG['patch_dir'], 'masks'),
            num_classes=CONFIG['num_classes']
        )

        val_dataset = SteverstalDataset(
            val_df,
            image_dir=os.path.join(CONFIG['patch_dir'], 'images'),
            mask_dir=os.path.join(CONFIG['patch_dir'], 'masks'),
            num_classes=CONFIG['num_classes']
        )

        train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)

        print(f"✅ Train samples: {len(train_dataset)}")
        print(f"✅ Val samples: {len(val_dataset)}")
        print(f"✅ Train batches: {len(train_loader)}")
        print(f"✅ Val batches: {len(val_loader)}")

        # ========== STEP 3: TRAINING ==========
        print("\n" + "=" * 70)
        print("STEP 3: TRAINING SEGMENTATION MODEL")
        print("=" * 70)

        model = UNet(in_channels=1, num_classes=CONFIG['num_classes'])
        model = model.to(CONFIG['device'])

        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n📊 Model Info:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Device: {CONFIG['device']}")

        trainer = SegmentationTrainer(model, CONFIG['device'], CONFIG['output_dir'])
        best_model_path = trainer.train(
            train_loader,
            val_loader,
            num_epochs=CONFIG['num_epochs'],
            learning_rate=CONFIG['learning_rate'],
            patience=10
        )

        trainer.plot_history(os.path.join(CONFIG['output_dir'], '02_training_history.png'))

        # ========== STEP 4: INFERENCE ==========
        print("\n" + "=" * 70)
        print("STEP 4: INFERENCE & VISUALIZATION")
        print("=" * 70)

        model.load_state_dict(torch.load(best_model_path))
        inferencer = SegmentationInference(model, CONFIG['device'])

        # Test on sample image
        sample_img_path = os.path.join(CONFIG['patch_dir'], 'images', train_df.iloc[0]['ImageId'])
        sample_img = cv2.imread(sample_img_path, cv2.IMREAD_GRAYSCALE)

        predictions = inferencer.predict(sample_img, threshold=0.5)
        inferencer.visualize_predictions(
            sample_img,
            predictions,
            os.path.join(CONFIG['output_dir'], '03_prediction_sample.png')
        )

        # ========== STEP 5: SAVE RESULTS ==========
        print("\n" + "=" * 70)
        print("STEP 5: SAVING RESULTS & CONFIGURATION")
        print("=" * 70)

        # Save config
        config_path = os.path.join(CONFIG['output_dir'], 'config.json')
        with open(config_path, 'w') as f:
            json.dump({k: str(v) if not isinstance(v, (int, float, bool, dict)) else v
                       for k, v in CONFIG.items()}, f, indent=4)

        print(f"✅ Configuration saved to {config_path}")
        print(f"✅ Best model saved to {best_model_path}")
        print(f"✅ All results saved to {CONFIG['output_dir']}")

        # ========== SUMMARY ==========
        print("\n" + "=" * 70)
        print("✨ PIPELINE COMPLETE!")
        print("=" * 70)
        print(f"\n📊 Summary:")
        print(f"   ✓ Preprocessed {len(df_patches)} patch records")
        print(f"   ✓ Trained U-Net for {len(trainer.history['train_loss'])} epochs")
        print(f"   ✓ Best validation Dice: {max(trainer.history['val_dice']):.4f}")
        print(f"   ✓ Best validation IoU: {max(trainer.history['val_iou']):.4f}")
        print(f"\n📁 Output files:")
        print(f"   - {CONFIG['output_dir']}/01_data_analysis.png")
        print(f"   - {CONFIG['output_dir']}/02_training_history.png")
        print(f"   - {CONFIG['output_dir']}/03_prediction_sample.png")
        print(f"   - {CONFIG['output_dir']}/best_model.pt")
        print(f"   - {CONFIG['output_dir']}/config.json")
        print(f"\n🎯 Next steps:")
        print(f"   1. Run inference on test set")
        print(f"   2. Generate predictions and save as .npy files")
        print(f"   3. Use KaggleSubmissionBuilder to convert to RLE format")
        print(f"   4. Submit to Kaggle competition")

        return CONFIG, trainer, model, train_loader, val_loader

    if __name__ == "__main__":
        config, trainer, model, train_loader, val_loader = main()