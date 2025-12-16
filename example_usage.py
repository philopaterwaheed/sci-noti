"""
Example Usage: Automated FER2013 Data Cleaning
Demonstrates how to use the automated cleaning pipeline
"""

import numpy as np
from fer2013_cleaner import FER2013Cleaner
from data_cleaner import FacialExpressionDataCleaner


def example_fer2013_quick_pipeline():
    """Example: Quick FER2013 download and cleaning"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Quick FER2013 Pipeline")
    print("="*60 + "\n")
    
    # Initialize cleaner
    cleaner = FER2013Cleaner(data_dir='./data')
    
    # Check if Kaggle is set up
    if not cleaner.check_kaggle_setup():
        print("\n⚠️  Note: Kaggle API not configured.")
        print("    This example shows the workflow, but won't download data.")
        print("    To actually download, set up Kaggle API credentials.")
        return
    
    # Run full pipeline
    # This will: download -> load -> clean -> enhance -> save
    images, labels = cleaner.full_pipeline(
        download=True,
        clean=True,
        enhance=True,
        save=True
    )
    
    if images is not None:
        print(f"\n✓ Ready for training!")
        print(f"  Images: {images.shape}")
        print(f"  Labels: {labels.shape}")
        print(f"  Saved to: ./data/fer2013/fer2013_cleaned.npz")


def example_manual_cleaning():
    """Example: Manual data cleaning with custom settings"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Manual Cleaning with Custom Settings")
    print("="*60 + "\n")
    
    # Create dummy data for demonstration
    print("Creating sample data (48x48 grayscale images)...")
    n_samples = 100
    images = np.random.randint(0, 255, (n_samples, 48, 48), dtype=np.uint8)
    labels = np.random.randint(0, 7, n_samples)
    
    print(f"Sample data: {images.shape}, labels: {labels.shape}")
    
    # Initialize cleaner with custom thresholds
    cleaner = FacialExpressionDataCleaner(
        blur_threshold=80,           # Higher = require sharper images
        brightness_range=(40, 220),  # Acceptable brightness range
        min_face_ratio=0.30,         # Minimum face size ratio
        outlier_std_multiplier=2.5   # Outlier detection sensitivity
    )
    
    # Clean the dataset (without face detection for this demo)
    clean_images, clean_labels = cleaner.clean_dataset(
        images=images,
        labels=labels,
        use_face_detection=False,  # Set True if MTCNN is installed
        use_embeddings=False,       # Set True if InsightFace is installed
        use_clahe=True,             # Apply CLAHE enhancement
        verbose=True
    )
    
    print(f"\n✓ Cleaning complete!")
    print(f"  Original: {len(images)} images")
    print(f"  Cleaned: {len(clean_images)} images")
    print(f"  Retention rate: {len(clean_images)/len(images)*100:.1f}%")


def example_load_cleaned_data():
    """Example: Load previously cleaned data"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Load Previously Cleaned Data")
    print("="*60 + "\n")
    
    cleaner = FER2013Cleaner(data_dir='./data')
    
    # Try to load cleaned data
    images, labels = cleaner.load_cleaned_data()
    
    if images is not None:
        print(f"\n✓ Loaded cleaned FER2013 data")
        print(f"  Images: {images.shape}")
        print(f"  Labels: {labels.shape}")
        print(f"\nReady for model training!")
    else:
        print("\n⚠️  No cleaned data found.")
        print("    Run example_fer2013_quick_pipeline() first to create it.")


def example_compare_cleaning_impact():
    """Example: Compare original vs cleaned data quality"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Cleaning Impact Analysis")
    print("="*60 + "\n")
    
    # Create sample data with various quality levels
    print("Creating sample dataset with quality issues...")
    
    good_images = np.random.randint(100, 150, (30, 48, 48), dtype=np.uint8)
    blurry_images = np.random.randint(50, 100, (20, 48, 48), dtype=np.uint8)
    dark_images = np.random.randint(0, 30, (20, 48, 48), dtype=np.uint8)
    bright_images = np.random.randint(230, 255, (20, 48, 48), dtype=np.uint8)
    low_contrast = np.full((10, 48, 48), 128, dtype=np.uint8)
    
    all_images = np.concatenate([
        good_images, blurry_images, dark_images, 
        bright_images, low_contrast
    ])
    all_labels = np.random.randint(0, 7, len(all_images))
    
    print(f"Total images: {len(all_images)}")
    print(f"  - Good quality: 30")
    print(f"  - Blurry: 20")
    print(f"  - Too dark: 20")
    print(f"  - Too bright: 20")
    print(f"  - Low contrast: 10")
    
    # Apply cleaning
    cleaner = FacialExpressionDataCleaner()
    clean_images, clean_labels = cleaner.clean_dataset(
        images=all_images,
        labels=all_labels,
        use_face_detection=False,
        use_embeddings=False,
        use_clahe=True,
        verbose=True
    )
    
    print(f"\n✓ Analysis complete")
    print(f"  Original: {len(all_images)} images")
    print(f"  After cleaning: {len(clean_images)} images")
    print(f"  Removed: {len(all_images) - len(clean_images)} low-quality images")
    print(f"  Quality improvement: {(1 - len(clean_images)/len(all_images))*100:.1f}% of poor quality data removed")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print(" "*10 + "FER2013 AUTOMATED CLEANING - USAGE EXAMPLES")
    print("="*70)
    
    examples = [
        ("Quick FER2013 Pipeline", example_fer2013_quick_pipeline),
        ("Manual Cleaning with Custom Settings", example_manual_cleaning),
        ("Load Previously Cleaned Data", example_load_cleaned_data),
        ("Cleaning Impact Analysis", example_compare_cleaning_impact)
    ]
    
    print("\nAvailable Examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\n" + "-"*70)
    
    # Run example 2 and 4 (don't require Kaggle)
    example_manual_cleaning()
    example_compare_cleaning_impact()
    
    print("\n" + "="*70)
    print("Examples complete!")
    print("\nTo run FER2013 download examples:")
    print("  1. Set up Kaggle API: https://www.kaggle.com/account")
    print("  2. Run: python fer2013_cleaner.py --full")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
