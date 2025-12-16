"""
Kaggle Data Fetcher for Facial Expression Recognition
Automates downloading and organizing high-quality datasets from Kaggle
"""

import os
import subprocess
import json
from pathlib import Path


class KaggleDataFetcher:
    """Automated Kaggle dataset downloader for facial expression recognition"""
    
    # High-quality facial expression datasets on Kaggle
    RECOMMENDED_DATASETS = {
        'fer2013': 'msambare/fer2013',
        'fer2013_plus': 'ashishpatel26/facial-expression-recognitionferchallenge',
        'ck_plus': 'shawon10/ckplus',
        'raf_db': 'shuvoalok/raf-db-dataset',
        'affectnet': 'tom99763/affectnet-for-facial-expression-recognition',
        'jaffe': 'sujaykapadnis/jaffe',
        'expw': 'shuvoalok/expw'
    }
    
    def __init__(self, download_dir='./data'):
        """
        Initialize Kaggle data fetcher
        
        Args:
            download_dir: Directory to download datasets to
        """
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
    def check_kaggle_setup(self):
        """Check if Kaggle API is properly configured"""
        try:
            # Check if kaggle command exists
            subprocess.run(['kaggle', '--version'], 
                          capture_output=True, check=True)
            
            # Check for API credentials
            kaggle_config = Path.home() / '.kaggle' / 'kaggle.json'
            if not kaggle_config.exists():
                print("⚠️  Kaggle API credentials not found!")
                print("    Please set up your Kaggle API credentials:")
                print("    1. Go to https://www.kaggle.com/account")
                print("    2. Create new API token (downloads kaggle.json)")
                print("    3. Place it at ~/.kaggle/kaggle.json")
                print("    4. Run: chmod 600 ~/.kaggle/kaggle.json")
                return False
                
            print("✓ Kaggle API is properly configured")
            return True
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️  Kaggle CLI not installed!")
            print("    Install with: pip install kaggle")
            return False
    
    def download_dataset(self, dataset_name, force=False):
        """
        Download a specific dataset from Kaggle
        
        Args:
            dataset_name: Key from RECOMMENDED_DATASETS or full dataset path
            force: Force re-download even if exists
            
        Returns:
            Path to downloaded dataset directory
        """
        # Get full dataset path
        if dataset_name in self.RECOMMENDED_DATASETS:
            dataset_path = self.RECOMMENDED_DATASETS[dataset_name]
        else:
            dataset_path = dataset_name
            
        # Create dataset directory
        dataset_dir = self.download_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if already downloaded
        if not force and any(dataset_dir.iterdir()):
            print(f"✓ Dataset '{dataset_name}' already exists at {dataset_dir}")
            return dataset_dir
            
        print(f"📥 Downloading '{dataset_name}' from Kaggle...")
        print(f"    Dataset: {dataset_path}")
        print(f"    Target: {dataset_dir}")
        
        try:
            # Download dataset
            subprocess.run([
                'kaggle', 'datasets', 'download',
                '-d', dataset_path,
                '-p', str(dataset_dir),
                '--unzip'
            ], check=True, capture_output=True, text=True)
            
            print(f"✓ Successfully downloaded '{dataset_name}'")
            return dataset_dir
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to download '{dataset_name}'")
            print(f"    Error: {e.stderr}")
            return None
    
    def download_all(self, datasets=None, force=False):
        """
        Download multiple datasets
        
        Args:
            datasets: List of dataset names to download (None = all recommended)
            force: Force re-download even if exists
            
        Returns:
            Dictionary mapping dataset names to their paths
        """
        if datasets is None:
            datasets = list(self.RECOMMENDED_DATASETS.keys())
            
        results = {}
        
        print(f"\n{'='*60}")
        print(f"Downloading {len(datasets)} datasets from Kaggle")
        print(f"{'='*60}\n")
        
        for dataset in datasets:
            path = self.download_dataset(dataset, force=force)
            if path:
                results[dataset] = path
            print()
            
        print(f"\n{'='*60}")
        print(f"Download Summary: {len(results)}/{len(datasets)} successful")
        print(f"{'='*60}\n")
        
        return results
    
    def get_dataset_info(self):
        """Get information about available datasets"""
        print("\n" + "="*60)
        print("Available High-Quality Facial Expression Datasets")
        print("="*60 + "\n")
        
        for name, path in self.RECOMMENDED_DATASETS.items():
            dataset_dir = self.download_dir / name
            status = "✓ Downloaded" if dataset_dir.exists() and any(dataset_dir.iterdir()) else "○ Not downloaded"
            print(f"{status} | {name:20s} | {path}")
        
        print("\n" + "="*60 + "\n")


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Download facial expression datasets from Kaggle'
    )
    parser.add_argument(
        '--datasets', 
        nargs='+',
        help='Specific datasets to download (default: all)'
    )
    parser.add_argument(
        '--dir',
        default='./data',
        help='Directory to download datasets to (default: ./data)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if exists'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available datasets'
    )
    
    args = parser.parse_args()
    
    fetcher = KaggleDataFetcher(download_dir=args.dir)
    
    if args.list:
        fetcher.get_dataset_info()
        return
    
    # Check Kaggle setup
    if not fetcher.check_kaggle_setup():
        return
    
    # Download datasets
    fetcher.download_all(datasets=args.datasets, force=args.force)


if __name__ == '__main__':
    main()
