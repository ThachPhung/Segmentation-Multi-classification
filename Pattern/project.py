"""
Severstal Steel Defect Detection - Complete Pipeline
1. Data Preprocessing & Cropping
2. U-Net Segmentation Model
3. Training with Loss Functions
4. Evaluation & Inference
5. Defect Classification
"""

import os
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
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score
from datetime import datetime
import json


# ============================================================================
# 1. RLE & PREPROCESSING UTILITIES
# ============================================================================

class RLEProcessor:
    """RLE encoding/decoding utility"""

    @staticmethod
    def rle_decode(mask_rle, shape=(256, 1600)):
        if pd.isna(mask_rle) or mask_rle == '':
            return np.zeros(shape, dtype=np.uint8)

        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths

        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1

        return img.reshape(shape, order='F')

    @staticmethod
    def rle_encode(mask):
        pixels = mask.T.flatten(order='F')
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]

        return ' '.join(str(x) for x in runs)


# ============================================================================
# 2. DATA LOADING & PREPROCESSING
# ============================================================================

class DataPreprocessor:
    """Load, crop, and prepare data"""

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
        df = pd.read_csv(self.csv_path)
        df[['ImageId', 'ClassId']] = df['ImageId_ClassId'].str.rsplit('_', 1, expand=True)
        df['ClassId'] = df['ClassId'].astype(int)
        df['HasDefect'] = df['EncodedPixels'].notna()
        return df

    def create_patches(self, df, save_images=True):
        """Convert 256x1600 images into 4x 256x400 patches"""
        print("\n" + "=" * 60)
        print("✂️  CREATING IMAGE PATCHES")
        print("=" * 60)

        new_rows = []
        unique_images = df['ImageId'].unique()

        for img_id in tqdm(unique_images, desc="Cropping images"):
            img_path = self.image_dir / img_id
            if not img_path.exists():
                continue

            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            img_data = df[df['ImageId'] == img_id]

            # Create 4 patches
            for patch_idx in range(4):
                start_col = patch_idx * self.stride
                end_col = start_col + self.stride

                patch_img = image[:, start_col:end_col].copy()
                patch_id = f"{img_id[:-4]}_patch{patch_idx}.jpg"

                if save_images:
                    patch_path = self.output_dir / 'images' / patch_id
                    patch_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(patch_path), patch_img)

                # Process masks for each class
                for _, row in img_data.iterrows():
                    if pd.notna(row['EncodedPixels']):
                        full_mask = RLEProcessor.rle_decode(row['EncodedPixels'], self.original_shape)
                        patch_mask = full_mask[:, start_col:end_col].copy()
                        cropped_rle = RLEProcessor.rle_encode(patch_mask)
                    else:
                        cropped_rle = ''

                    new_rows.append({
                        'ImageId': patch_id,
                        'ClassId': row['ClassId'],
                        'EncodedPixels': cropped_rle,
                        'HasDefect': cropped_rle != ''
                    })

        df_patches = pd.DataFrame(new_rows)
        print(f"✅ Created {len(df_patches)} patch records")
        print(f"✅ Unique patch images: {df_patches['ImageId'].nunique()}")

        return df_patches


# ============================================================================
# 3. PYTORCH DATASET
# ============================================================================

class SteverstalDataset(Dataset):
    """PyTorch Dataset for Severstal"""

    def __init__(self, df, image_dir, mask_dir=None, transform=None, num_classes=4):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.transform = transform
        self.num_classes = num_classes
        self.unique_images = df['ImageId'].unique()

    def __len__(self):
        return len(self.unique_images)

    def __getitem__(self, idx):
        img_id = self.unique_images[idx]

        # Load image
        img_path = self.image_dir / img_id
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Normalize image
        image = image.astype(np.float32) / 255.0

        # Load masks for all 4 classes
        masks = []
        img_data = self.df[self.df['ImageId'] == img_id]

        for class_id in range(1, self.num_classes + 1):
            class_row = img_data[img_data['ClassId'] == class_id]

            if not class_row.empty and class_row['HasDefect'].values[0]:
                rle = class_row['EncodedPixels'].values[0]
                mask = RLEProcessor.rle_decode(rle, image.shape)
            else:
                mask = np.zeros(image.shape, dtype=np.uint8)

            masks.append(mask.astype(np.float32))

        masks = np.stack(masks, axis=0)  # (num_classes, H, W)

        # Apply transforms
        if self.transform:
            # Apply augmentation
            pass

        # Convert to tensors
        image_tensor = torch.from_numpy(image).float().unsqueeze(0)  # (1, H, W)
        masks_tensor = torch.from_numpy(masks).float()  # (num_classes, H, W)

        return {
            'image': image_tensor,
            'masks': masks_tensor,
            'image_id': img_id
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

        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.middle = DoubleConv(512, 1024)

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
            'train_dice': [],
            'val_dice': [],
            'train_iou': [],
            'val_iou': []
        }

    def train_epoch(self, train_loader, criterion, optimizer):
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc="Training"):
            images = batch['image'].to(self.device)
            masks = batch['masks'].to(self.device)

            # Forward
            outputs = self.model(images)

            # Loss
            loss = 0
            for c in range(masks.shape[1]):
                loss += criterion(outputs[:, c:c + 1, :, :], masks[:, c:c + 1, :, :])
            loss = loss / masks.shape[1]

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        return avg_loss

    @torch.no_grad()
    def validate(self, val_loader, criterion):
        """Validate"""
        self.model.eval()
        total_loss = 0.0
        all_dices = []
        all_ious = []

        for batch in tqdm(val_loader, desc="Validating"):
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
                pred_c = preds[:, c].cpu().numpy().astype(np.uint8)
                mask_c = masks[:, c].cpu().numpy().astype(np.uint8)

                dice = self._dice_coefficient(pred_c, mask_c)
                iou = jaccard_score(mask_c.flatten(), pred_c.flatten(), zero_division=0)

                all_dices.append(dice)
                all_ious.append(iou)

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

    def train(self, train_loader, val_loader, num_epochs, learning_rate=1e-3):
        """Full training loop"""
        criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        best_val_loss = float('inf')
        best_model_path = self.output_dir / 'best_model.pt'

        print("\n" + "=" * 60)
        print("🚀 STARTING TRAINING")
        print("=" * 60)

        for epoch in range(num_epochs):
            print(f"\n📊 Epoch {epoch + 1}/{num_epochs}")

            # Train
            train_loss = self.train_epoch(train_loader, criterion, optimizer)

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
                torch.save(self.model.state_dict(), best_model_path)
                print(f"   ✅ Best model saved!")

            scheduler.step(val_loss)

        print("\n✨ Training complete!")
        return best_model_path

    def plot_history(self, output_path='training_history.png'):
        """Plot training history"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(self.history['train_loss'], label='Train')
        axes[0].plot(self.history['val_loss'], label='Val')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss')
        axes[0].legend()
        axes[0].grid()

        axes[1].plot(self.history['val_dice'])
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Dice Score')
        axes[1].set_title('Validation Dice Score')
        axes[1].grid()

        axes[2].plot(self.history['val_iou'])
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('IoU')
        axes[2].set_title('Validation IoU')
        axes[2].grid()

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Training history saved to {output_path}")
        plt.show()


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
        image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
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

    def visualize_predictions(self, image, masks, output_path='prediction.png'):
        """Visualize predictions"""
        fig, axes = plt.subplots(1, 5, figsize=(16, 3))

        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        for idx, (class_name, mask) in enumerate(masks.items(), 1):
            axes[idx].imshow(mask, cmap='hot')
            axes[idx].set_title(class_name)
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"✅ Visualization saved to {output_path}")
        plt.show()


# ============================================================================
# 8. MULTI-CLASS CLASSIFICATION
# ============================================================================

class DefectClassifier(nn.Module):
    """Multi-class classifier for defect types"""

    def __init__(self, num_classes=4, input_channels=5):
        super(DefectClassifier, self).__init__()

        # Takes: image + 4 segmentation masks = 5 channels
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, image, masks):
        """
        Args:
            image: (B, 1, H, W)
            masks: (B, 4, H, W)
        """
        x = torch.cat([image, masks], dim=1)  # (B, 5, H, W)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ============================================================================
# 9. MAIN EXECUTION
# ============================================================================

def main():
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

    print("\n" + "🚀 " * 25)
    print("SEVERSTAL STEEL DEFECT DETECTION - FULL PIPELINE")
    print("🚀 " * 25)

    # Step 1: Preprocess data
    print("\n" + "=" * 60)
    print("STEP 1: DATA PREPROCESSING")
    print("=" * 60)

    preprocessor = DataPreprocessor(
        csv_path=CONFIG['train_csv'],
        image_dir=CONFIG['train_images'],
        output_dir=CONFIG['patch_dir']
    )

    df = preprocessor.load_csv()
    df_patches = preprocessor.create_patches(df, save_images=True)
    df_patches.to_csv(os.path.join(CONFIG['patch_dir'], 'train_patches.csv'), index=False)

    # Step 2: Create DataLoaders
    print("\n" + "=" * 60)
    print("STEP 2: CREATING DATALOADERS")
    print("=" * 60)

    from sklearn.model_selection import train_test_split

    unique_images = df_patches['ImageId'].unique()
    train_images, val_images = train_test_split(
        unique_images, test_size=0.2, random_state=42
    )

    df_train = df_patches[df_patches['ImageId'].isin(train_images)]
    df_val = df_patches[df_patches['ImageId'].isin(val_images)]

    train_dataset = SteverstalDataset(
        df_train,
        image_dir=os.path.join(CONFIG['patch_dir'], 'images'),
        num_classes=CONFIG['num_classes']
    )

    val_dataset = SteverstalDataset(
        df_val,
        image_dir=os.path.join(CONFIG['patch_dir'], 'images'),
        num_classes=CONFIG['num_classes']
    )

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    print(f"✅ Train samples: {len(train_dataset)}")
    print(f"✅ Val samples: {len(val_dataset)}")

    # Step 3: Train Segmentation Model
    print("\n" + "=" * 60)
    print("STEP 3: TRAINING SEGMENTATION MODEL")
    print("=" * 60)

    model = UNet(in_channels=1, num_classes=CONFIG['num_classes'])
    model = model.to(CONFIG['device'])

    trainer = SegmentationTrainer(model, CONFIG['device'], CONFIG['output_dir'])
    best_model_path = trainer.train(
        train_loader,
        val_loader,
        num_epochs=CONFIG['num_epochs'],
        learning_rate=CONFIG['learning_rate']
    )

    trainer.plot_history(os.path.join(CONFIG['output_dir'], 'training_history.png'))

    # Step 4: Test Inference
    print("\n" + "=" * 60)
    print("STEP 4: TESTING SEGMENTATION INFERENCE")
    print("=" * 60)

    model.load_state_dict(torch.load(best_model_path))
    inferencer = SegmentationInference(model, CONFIG['device'])

    # Load sample image
    sample_img_path = os.path.join(CONFIG['patch_dir'], 'images', train_images[0])
    sample_img = cv2.imread(sample_img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

    predictions = inferencer.predict(sample_img, threshold=0.5)
    inferencer.visualize_predictions(
        sample_img,
        predictions,
        os.path.join(CONFIG['output_dir'], 'sample_prediction.png')
    )

    # Step 5: Save Configuration & Results
    print("\n" + "=" * 60)
    print("STEP 5: SAVING RESULTS")
    print("=" * 60)

    config_path = os.path.join(CONFIG['output_dir'], 'config.json')
    with open(config_path, 'w') as f:
        json.dump({k: str(v) if not isinstance(v, (int, float, bool, dict)) else v
                   for k, v in CONFIG.items()}, f, indent=4)

    print(f"✅ Configuration saved to {config_path}")
    print(f"✅ Best model saved to {best_model_path}")
    print(f"✅ All results saved to {CONFIG['output_dir']}")

    print("\n✨ FULL PIPELINE COMPLETE!")
    print("\n📊 Summary:")
    print(f"   - Preprocessed {len(df_patches)} patch records")
    print(f"   - Trained U-Net for {CONFIG['num_epochs']} epochs")
    print(f"   - Best validation IoU: {max(trainer.history['val_iou']):.4f}")
    print(f"   - Ready for classification step")

    return CONFIG, trainer, model, train_loader, val_loader


if __name__ == "__main__":
    config, trainer, model, train_loader, val_loader = main()