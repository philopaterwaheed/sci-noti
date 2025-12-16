# Facial Expression Recognition with Automated Data Cleaning

An advanced facial expression recognition system with automated data cleaning and Kaggle dataset integration.

## Features

### 🎯 FER2013 - Primary Dataset
**FER2013 is the core dataset for this project** - a widely-used benchmark for facial expression recognition containing 35,887 grayscale 48x48 pixel face images across 7 emotion categories.

### 🤖 Facial Expression Detection Model
- Deep learning model for 7 emotion classes: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- Advanced CNN architecture with residual blocks
- Data augmentation with Albumentations
- Test-time augmentation (TTA) for improved accuracy
- Class balancing and label smoothing

### 🧹 Automated Data Cleaning for FER2013
- **Blur Detection**: Laplacian variance-based sharpness filtering
- **Brightness Filtering**: Removes over/under-exposed images
- **Contrast Checking**: Identifies low-contrast images
- **Occlusion Detection**: Edge density analysis
- **Face Detection**: MTCNN-based face validation
- **Outlier Removal**: Embedding-based outlier detection with InsightFace
- **CLAHE Enhancement**: Adaptive histogram equalization

### 📥 Automated FER2013 Download from Kaggle
- One-command FER2013 download from Kaggle
- Automatic data preparation and cleaning
- Optional supplementary datasets: CK+, RAF-DB, AffectNet, JAFFE, ExpW, KDEF
- Easy dataset management and organization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/philopaterwaheed/sci-noti.git
cd sci-noti
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Kaggle API (for automated dataset downloading):
```bash
# 1. Go to https://www.kaggle.com/account
# 2. Create new API token (downloads kaggle.json)
# 3. Place it at ~/.kaggle/kaggle.json
# 4. Set permissions
chmod 600 ~/.kaggle/kaggle.json
```

## Quick Start

### Option 1: Quick FER2013 Setup (Recommended)

```bash
# Complete FER2013 download and cleaning in one command
python fer2013_cleaner.py --full

# Or step by step:
python fer2013_cleaner.py --download  # Download FER2013 from Kaggle
python fer2013_cleaner.py --clean     # Apply quality filtering
python fer2013_cleaner.py --enhance   # Apply CLAHE enhancement
python fer2013_cleaner.py --save      # Save cleaned data
```

### Option 2: Download Multiple Datasets from Kaggle

```bash
# List available datasets
python kaggle_data_fetcher.py --list

# Download FER2013 (primary) and supplementary datasets
python kaggle_data_fetcher.py --datasets fer2013 ck_plus raf_db --dir ./data

# Download all available datasets
python kaggle_data_fetcher.py --dir ./data
```

### Option 3: Use Data Cleaning in Python

```python
# For FER2013 specifically
from fer2013_cleaner import FER2013Cleaner

cleaner = FER2013Cleaner(data_dir='./data')
images, labels = cleaner.full_pipeline()

# For any dataset
from data_cleaner import FacialExpressionDataCleaner

cleaner = FacialExpressionDataCleaner(
    blur_threshold=80,
    brightness_range=(40, 220),
    min_face_ratio=0.30,
    outlier_std_multiplier=2.5
)

clean_images, clean_labels = cleaner.clean_dataset(
    images=X,
    labels=y,
    use_face_detection=True,
    use_embeddings=True,
    use_clahe=True,
    verbose=True
)
```

### Train the Model

Open the Jupyter notebook and run all cells:
```bash
jupyter notebook ultra_improved_emotion_model_\(6\).ipynb
```

## Project Structure

```
sci-noti/
├── ultra_improved_emotion_model_(6).ipynb  # Main training notebook with FER2013
├── fer2013_cleaner.py                      # FER2013-specific cleaner with Kaggle integration
├── kaggle_data_fetcher.py                  # General Kaggle dataset downloader
├── data_cleaner.py                         # Enhanced data cleaning module
├── requirements.txt                        # Python dependencies
├── README.md                               # This file
└── data/                                   # Downloaded datasets (created automatically)
    └── fer2013/                           # FER2013 primary dataset
        ├── fer2013.csv                    # Original FER2013 data
        └── fer2013_cleaned.npz            # Cleaned and enhanced data
```

## Data Cleaning Pipeline

The automated data cleaning pipeline is specifically optimized for FER2013 and consists of three main stages:

### Stage 1: Quality Filtering
- **Blur Detection**: Uses Laplacian variance to filter out blurry FER2013 images (threshold: 80)
- **Brightness Check**: Removes images that are too dark or too bright (range: 40-220)
- **Contrast Analysis**: Filters out low-contrast images (std dev < 20)
- **Occlusion Detection**: Identifies and removes occluded faces using edge density
- **Face Validation**: Ensures exactly one valid face per image (optional with MTCNN)

### Stage 2: Embedding-Based Outlier Removal
- Uses InsightFace to extract face embeddings from FER2013 images
- Calculates cosine distances within each emotion class
- Removes outliers based on statistical thresholds (2.5 std deviations)
- Preserves class distribution while improving data quality

### Stage 3: Enhancement
- Applies CLAHE for better contrast in FER2013 images
- Normalizes pixel values to [0, 1] range
- Prepares data for model training

### FER2013-Specific Optimizations
- Handles FER2013's 48x48 grayscale format natively
- Optimized thresholds based on FER2013 characteristics
- Preserves emotion class balance during cleaning
- Typically retains 70-85% of original FER2013 images with improved quality

## Available Kaggle Datasets

| Dataset | Description | Kaggle Path |
|---------|-------------|-------------|
| FER2013 | Classic facial expression dataset | msambare/fer2013 |
| FER2013+ | Enhanced version of FER2013 | ashishpatel26/facial-expression-recognitionferchallenge |
| CK+ | Extended Cohn-Kanade dataset | shawon10/ckplus |
| RAF-DB | Real-world Affective Faces Database | shuvoalok/raf-db-dataset |
| AffectNet | Large-scale facial expression dataset | tom99763/affectnet-for-facial-expression-recognition |
| JAFFE | Japanese Female Facial Expression | sujaykapadnis/jaffe |
| ExpW | Expression in-the-Wild | shuvoalok/expw |

## Model Performance

The model achieves improved accuracy through:
- **Cleaner Training Data**: Automated filtering removes low-quality samples
- **Multiple Datasets**: Combines data from various sources for better generalization
- **Advanced Architecture**: Residual blocks and batch normalization
- **Data Augmentation**: Horizontal flips, rotations, brightness/contrast adjustments
- **Class Balancing**: Weighted loss to handle class imbalance
- **Test-Time Augmentation**: Multiple predictions averaged for robustness

## Configuration

### Data Cleaner Settings

```python
cleaner = FacialExpressionDataCleaner(
    blur_threshold=80,           # Laplacian variance threshold (higher = sharper required)
    brightness_range=(40, 220),  # Acceptable brightness range
    min_face_ratio=0.30,         # Minimum face area ratio
    outlier_std_multiplier=2.5   # Standard deviations for outlier detection
)
```

### Model Hyperparameters

See the notebook for detailed model configuration:
- Learning rate: 3e-4
- Batch size: 64
- Epochs: 50 (with early stopping)
- Label smoothing: 0.1
- Data augmentation probability: 0.5-0.7

## Requirements

- Python 3.7+
- TensorFlow 2.10+
- OpenCV 4.5+
- MTCNN (face detection)
- InsightFace (face embeddings)
- Kaggle API (dataset downloading)

See `requirements.txt` for complete list.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Other Projects

This repository also contains:
- `the_scrper.py`: A web scraper for college news (Discord bot)
