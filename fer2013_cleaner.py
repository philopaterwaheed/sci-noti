"""
FER2013 Dataset Cleaner with Kaggle Integration
Automated downloading and cleaning of FER2013 dataset
"""

import os
import subprocess
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm


class FER2013Cleaner:
    """
    Automated FER2013 dataset downloader and cleaner
    FER2013 is the primary dataset for facial expression recognition
    """
    
    def __init__(self, data_dir='./data'):
        """
        Initialize FER2013 cleaner
        
        Args:
            data_dir: Directory to store FER2013 data
        """
        self.data_dir = Path(data_dir)
        self.fer2013_dir = self.data_dir / 'fer2013'
        self.fer2013_dir.mkdir(parents=True, exist_ok=True)
        
    def check_kaggle_setup(self):
        """Check if Kaggle API is configured"""
        try:
            subprocess.run(['kaggle', '--version'], 
                          capture_output=True, check=True)
            
            kaggle_config = Path.home() / '.kaggle' / 'kaggle.json'
            if not kaggle_config.exists():
                print("⚠️  Kaggle credentials not found!")
                print("    Setup: https://www.kaggle.com/account")
                return False
                
            print("✓ Kaggle API configured")
            return True
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️  Kaggle CLI not installed!")
            print("    Install: pip install kaggle")
            return False
    
    def download_fer2013(self, force=False):
        """
        Download FER2013 dataset from Kaggle
        
        Args:
            force: Force re-download even if exists
            
        Returns:
            Path to FER2013 CSV file
        """
        csv_path = self.fer2013_dir / 'fer2013.csv'
        
        # Check if already exists
        if csv_path.exists() and not force:
            print(f"✓ FER2013 already downloaded at {csv_path}")
            return csv_path
        
        print(f"\n{'='*60}")
        print("DOWNLOADING FER2013 FROM KAGGLE")
        print(f"{'='*60}")
        print("Dataset: msambare/fer2013")
        print(f"Target: {self.fer2013_dir}")
        
        try:
            # Download and unzip
            subprocess.run([
                'kaggle', 'datasets', 'download',
                '-d', 'msambare/fer2013',
                '-p', str(self.fer2013_dir),
                '--unzip'
            ], check=True, capture_output=True, text=True)
            
            # Find the CSV file (it might be in a subdirectory)
            csv_files = list(self.fer2013_dir.rglob('*.csv'))
            if csv_files:
                # Move to main directory if needed
                if csv_files[0] != csv_path:
                    csv_files[0].rename(csv_path)
                print(f"✓ FER2013 downloaded successfully")
                print(f"  Location: {csv_path}")
                return csv_path
            else:
                print("❌ CSV file not found after download")
                return None
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to download FER2013")
            print(f"    Error: {e.stderr if hasattr(e, 'stderr') else str(e)}")
            return None
    
    def load_fer2013(self, csv_path=None):
        """
        Load FER2013 dataset from CSV
        
        Args:
            csv_path: Path to FER2013 CSV (None = auto-detect)
            
        Returns:
            Tuple of (dataframe, image_array, labels_array)
        """
        if csv_path is None:
            csv_path = self.fer2013_dir / 'fer2013.csv'
        
        if not Path(csv_path).exists():
            print(f"❌ FER2013 CSV not found at {csv_path}")
            print("   Try downloading with: download_fer2013()")
            return None, None, None
        
        print(f"\n{'='*60}")
        print(f"LOADING FER2013")
        print(f"{'='*60}")
        print(f"Source: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Handle column naming variations
        if ' pixels' in df.columns:
            df.rename(columns={' pixels': 'pixels'}, inplace=True)
        if ' emotion' in df.columns:
            df.rename(columns={' emotion': 'emotion'}, inplace=True)
        
        print(f"✓ Loaded {len(df)} images")
        print(f"  Columns: {list(df.columns)}")
        print(f"\nEmotion distribution:")
        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        for i, label in enumerate(emotion_labels):
            count = (df['emotion'] == i).sum()
            print(f"  {label:10s}: {count:5d} ({count/len(df)*100:.1f}%)")
        
        # Convert to arrays
        print("\nConverting to image arrays...")
        images = []
        labels = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            try:
                # Parse pixel string
                pixels = np.fromstring(row['pixels'], dtype=np.uint8, sep=' ')
                img = pixels.reshape(48, 48)
                images.append(img)
                labels.append(int(row['emotion']))
            except:
                continue
        
        images = np.array(images)
        labels = np.array(labels)
        
        print(f"\n✓ FER2013 ready")
        print(f"  Images shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        
        return df, images, labels
    
    def apply_basic_cleaning(self, images, labels, 
                            blur_threshold=80,
                            brightness_range=(40, 220)):
        """
        Apply basic quality filters to FER2013
        
        Args:
            images: Array of images (H, W) or (N, H, W)
            labels: Array of labels
            blur_threshold: Minimum Laplacian variance
            brightness_range: (min, max) brightness
            
        Returns:
            Tuple of (clean_images, clean_labels)
        """
        print(f"\n{'='*60}")
        print("CLEANING FER2013 DATASET")
        print(f"{'='*60}")
        print(f"Settings:")
        print(f"  - Blur threshold: {blur_threshold}")
        print(f"  - Brightness range: {brightness_range}")
        
        clean_images = []
        clean_labels = []
        
        print("\nApplying quality filters...")
        for img, label in tqdm(zip(images, labels), total=len(images), desc="Filtering"):
            try:
                # Blur check
                laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
                if laplacian_var <= blur_threshold:
                    continue
                
                # Brightness check
                mean_brightness = img.mean()
                if not (brightness_range[0] < mean_brightness < brightness_range[1]):
                    continue
                
                # Contrast check
                if np.std(img) < 20:
                    continue
                
                clean_images.append(img)
                clean_labels.append(label)
                
            except Exception as e:
                continue
        
        clean_images = np.array(clean_images)
        clean_labels = np.array(clean_labels)
        
        kept_pct = len(clean_images) / len(images) * 100
        print(f"\n✓ Cleaning complete")
        print(f"  Original: {len(images)} images")
        print(f"  Cleaned: {len(clean_images)} images")
        print(f"  Kept: {kept_pct:.1f}%")
        
        return clean_images, clean_labels
    
    def apply_clahe(self, images):
        """
        Apply CLAHE enhancement to images
        
        Args:
            images: Array of grayscale images
            
        Returns:
            Enhanced images
        """
        print("\nApplying CLAHE enhancement...")
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        enhanced = []
        for img in tqdm(images, desc="Enhancing"):
            enhanced.append(clahe.apply(img.astype(np.uint8)))
        
        return np.array(enhanced)
    
    def save_cleaned_data(self, images, labels, output_path=None):
        """
        Save cleaned FER2013 data
        
        Args:
            images: Cleaned image array
            labels: Label array
            output_path: Where to save (None = auto)
        """
        if output_path is None:
            output_path = self.fer2013_dir / 'fer2013_cleaned.npz'
        
        print(f"\nSaving cleaned data to {output_path}...")
        np.savez_compressed(output_path, images=images, labels=labels)
        print(f"✓ Saved successfully")
        print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    def load_cleaned_data(self, input_path=None):
        """
        Load previously cleaned FER2013 data
        
        Args:
            input_path: Path to cleaned data (None = auto)
            
        Returns:
            Tuple of (images, labels)
        """
        if input_path is None:
            input_path = self.fer2013_dir / 'fer2013_cleaned.npz'
        
        if not Path(input_path).exists():
            print(f"❌ Cleaned data not found at {input_path}")
            return None, None
        
        print(f"Loading cleaned data from {input_path}...")
        data = np.load(input_path)
        images = data['images']
        labels = data['labels']
        
        print(f"✓ Loaded cleaned FER2013")
        print(f"  Images: {images.shape}")
        print(f"  Labels: {labels.shape}")
        
        return images, labels
    
    def full_pipeline(self, download=True, clean=True, enhance=True, save=True):
        """
        Run complete FER2013 preparation pipeline
        
        Args:
            download: Download from Kaggle if needed
            clean: Apply quality filtering
            enhance: Apply CLAHE enhancement
            save: Save cleaned data
            
        Returns:
            Tuple of (images, labels)
        """
        print("\n" + "="*60)
        print("FER2013 FULL PREPARATION PIPELINE")
        print("="*60)
        print(f"Steps: Download={download}, Clean={clean}, Enhance={enhance}, Save={save}")
        
        # Step 1: Download
        if download:
            csv_path = self.download_fer2013()
            if csv_path is None:
                return None, None
        else:
            csv_path = self.fer2013_dir / 'fer2013.csv'
        
        # Step 2: Load
        df, images, labels = self.load_fer2013(csv_path)
        if images is None:
            return None, None
        
        # Step 3: Clean
        if clean:
            images, labels = self.apply_basic_cleaning(images, labels)
        
        # Step 4: Enhance
        if enhance:
            images = self.apply_clahe(images)
        
        # Step 5: Save
        if save:
            self.save_cleaned_data(images, labels)
        
        print("\n" + "="*60)
        print("✓ PIPELINE COMPLETE")
        print("="*60)
        print(f"Final dataset: {images.shape}")
        
        return images, labels


def main():
    """Command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Download and clean FER2013 dataset from Kaggle'
    )
    parser.add_argument('--dir', default='./data',
                       help='Data directory (default: ./data)')
    parser.add_argument('--download', action='store_true',
                       help='Download FER2013 from Kaggle')
    parser.add_argument('--clean', action='store_true',
                       help='Apply cleaning filters')
    parser.add_argument('--enhance', action='store_true',
                       help='Apply CLAHE enhancement')
    parser.add_argument('--save', action='store_true',
                       help='Save cleaned data')
    parser.add_argument('--full', action='store_true',
                       help='Run full pipeline (download+clean+enhance+save)')
    
    args = parser.parse_args()
    
    cleaner = FER2013Cleaner(data_dir=args.dir)
    
    if args.full:
        # Check Kaggle setup first
        if not cleaner.check_kaggle_setup():
            print("\n⚠️  Please set up Kaggle API first")
            return
        
        # Run full pipeline
        cleaner.full_pipeline(
            download=True,
            clean=True,
            enhance=True,
            save=True
        )
    else:
        # Individual steps
        if args.download:
            if not cleaner.check_kaggle_setup():
                return
            cleaner.download_fer2013()
        
        if args.clean or args.enhance or args.save:
            df, images, labels = cleaner.load_fer2013()
            
            if args.clean:
                images, labels = cleaner.apply_basic_cleaning(images, labels)
            
            if args.enhance:
                images = cleaner.apply_clahe(images)
            
            if args.save:
                cleaner.save_cleaned_data(images, labels)


if __name__ == '__main__':
    main()
