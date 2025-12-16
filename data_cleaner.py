"""
Enhanced Data Cleaning Module for Facial Expression Recognition
Provides automated quality filtering and preprocessing
"""

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_distances


class FacialExpressionDataCleaner:
    """Comprehensive data cleaning for facial expression datasets"""
    
    def __init__(self, 
                 blur_threshold=80,
                 brightness_range=(40, 220),
                 min_face_ratio=0.30,
                 outlier_std_multiplier=2.5):
        """
        Initialize data cleaner with quality thresholds
        
        Args:
            blur_threshold: Minimum Laplacian variance for sharpness
            brightness_range: (min, max) acceptable brightness values
            min_face_ratio: Minimum ratio of face area to image area
            outlier_std_multiplier: Standard deviations for outlier detection
        """
        self.blur_threshold = blur_threshold
        self.brightness_range = brightness_range
        self.min_face_ratio = min_face_ratio
        self.outlier_std_multiplier = outlier_std_multiplier
        
        self.face_detector = None
        self.embedding_model = None
        
    def init_face_detector(self):
        """Initialize MTCNN face detector (lazy loading)"""
        if self.face_detector is not None:
            return self.face_detector
            
        try:
            from mtcnn import MTCNN
            self.face_detector = MTCNN()
            print("✓ MTCNN face detector loaded")
        except ImportError:
            print("⚠️  MTCNN not available. Install with: pip install mtcnn")
            self.face_detector = None
            
        return self.face_detector
    
    def init_embedding_model(self):
        """Initialize InsightFace embedding model (lazy loading)"""
        if self.embedding_model is not None:
            return self.embedding_model
            
        try:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=0)
            self.embedding_model = app
            print("✓ InsightFace embedding model loaded")
        except ImportError:
            print("⚠️  InsightFace not available. Install with: pip install insightface")
            self.embedding_model = None
        except Exception as e:
            print(f"⚠️  Could not load InsightFace: {e}")
            self.embedding_model = None
            
        return self.embedding_model
    
    def is_blur_ok(self, img):
        """
        Check if image has acceptable sharpness using Laplacian variance
        
        Args:
            img: Grayscale image
            
        Returns:
            True if image is sharp enough
        """
        laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
        return laplacian_var > self.blur_threshold
    
    def is_brightness_ok(self, img):
        """
        Check if image has acceptable brightness
        
        Args:
            img: Grayscale image
            
        Returns:
            True if brightness is in acceptable range
        """
        mean_brightness = img.mean()
        return self.brightness_range[0] < mean_brightness < self.brightness_range[1]
    
    def has_contrast_issues(self, img):
        """
        Check if image has low contrast
        
        Args:
            img: Grayscale image
            
        Returns:
            True if contrast is too low
        """
        std_dev = np.std(img)
        return std_dev < 20  # Low contrast threshold
    
    def detect_occlusions(self, img):
        """
        Simple occlusion detection using edge density
        
        Args:
            img: Grayscale image
            
        Returns:
            True if potential occlusions detected
        """
        edges = cv2.Canny(img, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Too few or too many edges might indicate occlusion
        return edge_density < 0.05 or edge_density > 0.30
    
    def has_valid_face(self, img):
        """
        Check if image contains exactly one valid face
        
        Args:
            img: Grayscale image
            
        Returns:
            True if exactly one valid face detected
        """
        if self.face_detector is None:
            return True  # Skip if detector not available
            
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        faces = self.face_detector.detect_faces(rgb)
        
        # Must have exactly one face
        if len(faces) != 1:
            return False
        
        # Check face size
        x, y, w, h = faces[0]['box']
        face_area = w * h
        img_area = img.shape[0] * img.shape[1]
        
        return (face_area / img_area) > self.min_face_ratio
    
    def get_face_embedding(self, img):
        """
        Extract face embedding from image
        
        Args:
            img: Grayscale image
            
        Returns:
            Face embedding vector or None
        """
        if self.embedding_model is None:
            return None
            
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        faces = self.embedding_model.get(rgb)
        
        if len(faces) != 1:
            return None
            
        return faces[0].embedding
    
    def remove_embedding_outliers(self, df_with_embeddings):
        """
        Remove outliers based on embedding distances per class
        
        Args:
            df_with_embeddings: DataFrame with 'embedding' and 'emotion' columns
            
        Returns:
            Cleaned DataFrame
        """
        def remove_outliers_per_class(group):
            if len(group) < 10:
                return group
            
            E = np.stack(group['embedding'].values)
            center = E.mean(axis=0)
            distances = cosine_distances(E, center.reshape(1, -1)).flatten()
            threshold = distances.mean() + self.outlier_std_multiplier * distances.std()
            
            return group[distances < threshold]
        
        cleaned = df_with_embeddings.groupby('emotion', group_keys=False).apply(
            remove_outliers_per_class
        )
        
        return cleaned.reset_index(drop=True)
    
    def apply_clahe(self, img):
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        
        Args:
            img: Grayscale image
            
        Returns:
            Enhanced image
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img)
    
    def clean_dataset(self, 
                     images, 
                     labels,
                     use_face_detection=True,
                     use_embeddings=True,
                     use_clahe=True,
                     verbose=True):
        """
        Clean a dataset with comprehensive quality filters
        
        Args:
            images: List or array of images
            labels: List or array of emotion labels
            use_face_detection: Whether to use face detection
            use_embeddings: Whether to use embedding-based outlier removal
            use_clahe: Whether to apply CLAHE enhancement
            verbose: Whether to print progress
            
        Returns:
            Tuple of (cleaned_images, cleaned_labels)
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Cleaning dataset with {len(images)} images")
            print(f"{'='*60}")
            print(f"Settings:")
            print(f"  - Face detection: {use_face_detection}")
            print(f"  - Embedding outliers: {use_embeddings}")
            print(f"  - CLAHE enhancement: {use_clahe}")
            print(f"  - Blur threshold: {self.blur_threshold}")
            print(f"  - Brightness range: {self.brightness_range}")
        
        # Initialize detectors if needed
        if use_face_detection:
            self.init_face_detector()
        if use_embeddings:
            self.init_embedding_model()
        
        # Step 1: Quality filtering
        if verbose:
            print(f"\n[Step 1/3] Applying quality filters...")
        
        clean_images = []
        clean_labels = []
        embeddings = []
        
        iterator = tqdm(zip(images, labels), total=len(images), desc="Filtering") if verbose else zip(images, labels)
        
        for img, label in iterator:
            try:
                # Ensure grayscale
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                
                # Quality checks
                if not self.is_blur_ok(img):
                    continue
                
                if not self.is_brightness_ok(img):
                    continue
                
                if self.has_contrast_issues(img):
                    continue
                
                if self.detect_occlusions(img):
                    continue
                
                # Face detection
                if use_face_detection and not self.has_valid_face(img):
                    continue
                
                # Extract embedding
                if use_embeddings:
                    emb = self.get_face_embedding(img)
                    if emb is None:
                        continue
                    embeddings.append(emb)
                
                # Apply CLAHE
                if use_clahe:
                    img = self.apply_clahe(img)
                
                clean_images.append(img)
                clean_labels.append(label)
                
            except Exception as e:
                if verbose:
                    print(f"Error processing image: {e}")
                continue
        
        if verbose:
            kept_pct = len(clean_images) / len(images) * 100
            print(f"After quality filtering: {len(clean_images)} images ({kept_pct:.1f}%)")
        
        # Step 2: Embedding-based outlier removal
        if use_embeddings and len(embeddings) > 0:
            if verbose:
                print(f"\n[Step 2/3] Removing embedding outliers...")
            
            df = pd.DataFrame({
                'image_idx': range(len(clean_images)),
                'emotion': clean_labels,
                'embedding': embeddings
            })
            
            cleaned_df = self.remove_embedding_outliers(df)
            
            # Filter images and labels
            kept_indices = cleaned_df['image_idx'].values
            clean_images = [clean_images[i] for i in kept_indices]
            clean_labels = [clean_labels[i] for i in kept_indices]
            
            if verbose:
                kept_pct = len(clean_images) / len(images) * 100
                print(f"After outlier removal: {len(clean_images)} images ({kept_pct:.1f}%)")
        
        # Step 3: Final conversion
        if verbose:
            print(f"\n[Step 3/3] Converting to arrays...")
        
        clean_images = np.array(clean_images)
        clean_labels = np.array(clean_labels)
        
        if verbose:
            print(f"\n✓ Cleaning complete!")
            print(f"  Final size: {clean_images.shape}")
            print(f"  Kept: {len(clean_images)}/{len(images)} images ({len(clean_images)/len(images)*100:.1f}%)\n")
        
        return clean_images, clean_labels


def main():
    """Example usage"""
    print("Data Cleaner Module")
    print("=" * 60)
    print("This module provides automated data cleaning for facial")
    print("expression recognition datasets.")
    print()
    print("Features:")
    print("  - Blur detection (Laplacian variance)")
    print("  - Brightness filtering")
    print("  - Contrast checking")
    print("  - Occlusion detection")
    print("  - Face detection (MTCNN)")
    print("  - Embedding-based outlier removal (InsightFace)")
    print("  - CLAHE enhancement")
    print()
    print("Usage:")
    print("  from data_cleaner import FacialExpressionDataCleaner")
    print("  cleaner = FacialExpressionDataCleaner()")
    print("  clean_X, clean_y = cleaner.clean_dataset(X, y)")
    print("=" * 60)


if __name__ == '__main__':
    main()
