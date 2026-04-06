"""
SEVERSTAL - COMPLETE PROJECT WITH ALL IMAGES (DEFECT + NO-DEFECT)
Xử lý cả những ảnh không có lỗi
"""

import os
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# 1. BUILD COMPLETE DATAFRAME - Includes images without defects
# ============================================================================

class CompleteDatasetBuilder:
    """Build complete dataset including images without defects"""

    def __init__(self, csv_path, image_dir, original_shape=(256, 1600)):
        self.csv_path = csv_path
        self.image_dir = Path(image_dir)
        self.original_shape = original_shape

    def build_complete_df(self):
        """
        Build complete dataframe with ALL images:
        - Images từ CSV (có defect)
        - Images trong folder nhưng không có trong CSV (không có defect)
        """
        print("\n" + "=" * 70)
        print("🔍 BUILDING COMPLETE DATASET")
        print("=" * 70)

        # Load CSV (chỉ ảnh có lỗi)
        df_csv = pd.read_csv(self.csv_path)
        df_csv['HasDefect'] = df_csv['EncodedPixels'].notna()

        images_with_defect = set(df_csv['ImageId'].unique())

        print(f"✅ Images từ CSV (có defect): {len(images_with_defect)}")

        # Lấy tất cả ảnh trong folder
        all_images = set([f for f in os.listdir(self.image_dir) if f.endswith(('.jpg', '.png'))])

        print(f"✅ Tất cả ảnh trong folder: {len(all_images)}")

        # Ảnh không có trong CSV (không có defect)
        images_without_defect = all_images - images_with_defect

        print(f"✅ Images không có defect (ko trong CSV): {len(images_without_defect)}")

        # Xây dựng complete dataframe
        rows = []

        # 1. Thêm tất cả records từ CSV (ảnh có defect)
        for _, row in df_csv.iterrows():
            rows.append(row)

        # 2. Thêm ảnh không có defect - tạo 4 rows (mỗi class 1 row)
        for img_id in images_without_defect:
            for class_id in range(1, 5):
                rows.append({
                    'ImageId_ClassId': f"{img_id}_{class_id}",
                    'ImageId': img_id,
                    'ClassId': class_id,
                    'EncodedPixels': '',  # Không có lỗi = RLE rỗng
                    'HasDefect': False
                })

        df_complete = pd.DataFrame(rows)

        print(f"\n✅ Complete dataframe:")
        print(f"   - Total records: {len(df_complete)}")
        print(f"   - Unique images: {df_complete['ImageId'].nunique()}")
        print(f"   - Images with defects: {df_complete[df_complete['HasDefect']].groupby('ImageId').ngroups}")
        print(f"   - Images without defects: {len(images_without_defect)}")

        # Thống kê
        print(f"\n📊 Class distribution:")
        for cls in range(1, 5):
            cls_data = df_complete[df_complete['ClassId'] == cls]
            defect_count = cls_data['HasDefect'].sum()
            total_count = len(cls_data)
            pct = (defect_count / total_count) * 100
            print(f"   Class {cls}: {int(defect_count):5} defects / {total_count:5} total ({pct:5.1f}%)")

        return df_complete


# ============================================================================
# 2. RLE PROCESSOR (giữ nguyên)
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
        """Encode binary mask to RLE string"""
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
                if pd.notna(rle_value) and rle_value != '':
                    mask[:, :, class_id - 1] = RLEProcessor.rle_decode(rle_value)

        return mask


# ============================================================================
# 3. DATA PREPROCESSOR - Handle all images
# ============================================================================

class DataPreprocessor:
    """Load, crop, and prepare data - including no-defect images"""

    def __init__(self, csv_path, image_dir, output_dir, patch_size=256, stride=400):
        self.csv_path = csv_path
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.patch_size = patch_size
        self.stride = stride
        self.original_shape = (256, 1600)
        self.patch_shape = (256, 400)

    def analyze_data(self, df):
        """Analyze dataset statistics"""
        print("\n" + "=" * 70)
        print("📊 DATA ANALYSIS")
        print("=" * 70)

        # Class distribution
        class_counts = df.groupby('ClassId')['HasDefect'].sum()
        class_names = {1: 'Crazing', 2: 'Inclusion', 3: 'Patches', 4: 'Scratches'}

        print("\n📈 Defects per class:")
        for cls in sorted(df['ClassId'].unique()):
            count = class_counts[cls]
            name = class_names.get(cls, f"Class {cls}")
            total = len(df[df['ClassId'] == cls])
            pct = (count / total) * 100
            print(f"   Class {cls} ({name:12}): {int(count):5} defects / {total:5} total ({pct:5.1f}%)")

        # Images statistics
        imgs_with_defect = df[df['HasDefect']].groupby('ImageId').size()
        imgs_without_defect = df['ImageId'].nunique() - len(imgs_with_defect)

        print(f"\n📸 Image statistics:")
        print(f"   Total images: {df['ImageId'].nunique()}")
        print(f"   With defect: {len(imgs_with_defect)} ({len(imgs_with_defect) / df['ImageId'].nunique() * 100:.1f}%)")
        print(f"   Without defect: {imgs_without_defect} ({imgs_without_defect / df['ImageId'].nunique() * 100:.1f}%)")

        # Visualize
        self._plot_analysis(df, class_names, class_counts, imgs_with_defect, imgs_without_defect)

    def _plot_analysis(self, df, class_names, class_counts, imgs_with, imgs_without):
        """Plot data analysis"""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle('Severstal Dataset Analysis (ALL Images)', fontsize=14, fontweight='bold')

        # Plot 1: Class distribution
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        axes[0].bar(range(1, 5), class_counts.values, color=colors, alpha=0.8, edgecolor='black')
        axes[0].set_xlabel('Defect Class')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Defects per Class (Imbalanced!)')
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
        axes[1].set_title('Class Distribution (Imbalanced!)')
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        # Plot 3: Images with/without defects
        axes[2].bar(['With Defect', 'Without Defect'], [len(imgs_with), len(imgs_without)],
                    color=['#FF6B6B', '#90EE90'], alpha=0.8, edgecolor='black')
        axes[2].set_ylabel('Count')
        axes[2].set_title('Images with/without Defects (Balanced!)')
        axes[2].grid(axis='y', alpha=0.3)
        for i, v in enumerate([len(imgs_with), len(imgs_without)]):
            axes[2].text(i, v, str(v), ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        Path('./outputs').mkdir(parents=True, exist_ok=True)
        plt.savefig('./outputs/01_data_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n✅ Analysis plot saved to './outputs/01_data_analysis.png'")
        plt.close()

    def create_patches(self, df, save_images=True, save_masks=True):
        """Convert 256×1600 images into 4× 256×400 patches"""
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
                print(f"⚠️  Image not found: {img_id}")
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
        print(f"✅ Patch statistics:")
        print(f"   - Patches with defects: {df_patches['HasDefect'].sum()}")
        print(f"   - Patches without defects: {(~df_patches['HasDefect']).sum()}")
        print(f"   - Defect balance: {df_patches['HasDefect'].sum() / len(df_patches) * 100:.1f}%")

        return df_patches


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Step 1: Build complete dataframe
    dataset_builder = CompleteDatasetBuilder(
        csv_path='./data/train.csv',
        image_dir='./data/train_images'
    )
    df_complete = dataset_builder.build_complete_df()

    # Step 2: Preprocess and create patches
    preprocessor = DataPreprocessor(
        csv_path='./data/train.csv',
        image_dir='./data/train_images',
        output_dir='./data/patches'
    )

    preprocessor.analyze_data(df_complete)
    df_patches = preprocessor.create_patches(df_complete, save_images=True, save_masks=True)

    # Save complete CSV
    df_complete.to_csv('./data/train_complete.csv', index=False)
    df_patches.to_csv('./data/patches/train_patches.csv', index=False)

    print("\n" + "=" * 70)
    print("✨ COMPLETE!")
    print("=" * 70)
    print(f"✅ Complete dataset saved to './data/train_complete.csv'")
    print(f"✅ Patch dataset saved to './data/patches/train_patches.csv'")
    print(f"✅ Total training samples: {len(df_patches)}")
    print(f"   - With defects: {df_patches['HasDefect'].sum()}")
    print(f"   - Without defects: {(~df_patches['HasDefect']).sum()}")