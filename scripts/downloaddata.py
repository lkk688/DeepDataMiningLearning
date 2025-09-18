
import argparse
import os
import urllib.request
import urllib.error
from pathlib import Path
from zipfile import ZipFile


def download_file(url, filepath, chunk_size=8192):
    """Download a file from URL to filepath"""
    try:
        print(f"Downloading {url}...")
        with urllib.request.urlopen(url) as response:
            with open(filepath, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def extract_file(filepath, extract_dir):
    """Extract zip/tar files"""
    try:
        if filepath.suffix == '.zip':
            print(f"Extracting {filepath}...")
            with ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        elif filepath.suffix in ['.tgz', '.gz']:
            print(f"Extracting {filepath}...")
            import tarfile
            with tarfile.open(filepath, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_dir)
        return True
    except Exception as e:
        print(f"Error extracting {filepath}: {e}")
        return False


def downloadurls(download_dir, urls):
    """Download and extract multiple URLs"""
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    
    for url in urls:
        filename = Path(url).name
        filepath = download_dir / filename
        
        # Download file
        if download_file(url, filepath):
            # Extract if it's an archive
            if filepath.suffix in ['.zip', '.tgz', '.gz']:
                extract_file(filepath, download_dir)
                # Optionally remove the archive after extraction
                # filepath.unlink()


# Dataset configurations
DATASETS = {
    'visdrone': {
        'name': 'VisDrone2019',
        'description': 'VisDrone2019 object detection dataset',
        'urls': [
            'https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-train.zip',
            'https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-val.zip',
            'https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-test-dev.zip',
            'https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-test-challenge.zip'
        ],
        'default_dir': '/data/cmpe249-fa23/VisDrone'
    },
    'coco': {
        'name': 'COCO Dataset',
        'description': 'COCO 2017 object detection dataset',
        'urls': [
            'http://images.cocodataset.org/zips/train2017.zip',
            'http://images.cocodataset.org/zips/val2017.zip',
            'http://images.cocodataset.org/zips/test2017.zip',
            'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
        ],
        'default_dir': '/data/cmpe249-fa23/COCO'
    },
    'coco128': {
        'name': 'COCO128 (Sample)',
        'description': 'COCO128 sample dataset for quick testing',
        'urls': [
            'https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128_with_yaml.zip'
        ],
        'default_dir': '/data/cmpe249-fa23/COCO128'
    },
    'kitti': {
        'name': 'KITTI Dataset',
        'description': 'KITTI object detection dataset (images, labels, calibration, velodyne)',
        'urls': [
            'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip',
            'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip',
            'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip',
            'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip'
        ],
        'default_dir': '/data/cmpe249-fa23/KITTI',
        'organize': True,
        'components': ['images', 'labels', 'calib', 'velodyne']
    },
    'nuscenes-mini': {
        'name': 'nuScenes Mini',
        'description': 'nuScenes mini dataset (10 scenes)',
        'urls': [
            'https://www.nuscenes.org/data/v1.0-mini.tgz'
        ],
        'default_dir': '/data/cmpe249-fa23/nuScenes-mini'
    },
    'nuscenes': {
        'name': 'nuScenes Full',
        'description': 'nuScenes full dataset (1000 scenes) - Very large download!',
        'urls': [
            'https://www.nuscenes.org/data/v1.0-trainval_meta.tgz',
            'https://www.nuscenes.org/data/v1.0-trainval01_blobs.tgz',
            'https://www.nuscenes.org/data/v1.0-trainval02_blobs.tgz',
            'https://www.nuscenes.org/data/v1.0-trainval03_blobs.tgz',
            'https://www.nuscenes.org/data/v1.0-trainval04_blobs.tgz',
            'https://www.nuscenes.org/data/v1.0-trainval05_blobs.tgz',
            'https://www.nuscenes.org/data/v1.0-trainval06_blobs.tgz',
            'https://www.nuscenes.org/data/v1.0-trainval07_blobs.tgz',
            'https://www.nuscenes.org/data/v1.0-trainval08_blobs.tgz',
            'https://www.nuscenes.org/data/v1.0-trainval09_blobs.tgz',
            'https://www.nuscenes.org/data/v1.0-trainval10_blobs.tgz'
        ],
        'default_dir': '/data/cmpe249-fa23/nuScenes'
    },
    'nu-images-mini': {
        'name': 'nuImages Mini',
        'description': 'nuImages mini dataset for 2D object detection',
        'urls': [
            'https://www.nuscenes.org/data/nuimages-v1.0-mini.tgz'
        ],
        'default_dir': '/data/cmpe249-fa23/nuImages-mini'
    }
}


def list_datasets():
    """List all available datasets"""
    print("\nAvailable datasets:")
    print("-" * 50)
    for key, dataset in DATASETS.items():
        status = " (Manual download required)" if dataset.get('manual_download') else ""
        print(f"{key:15} - {dataset['name']}{status}")
        print(f"{'':15}   {dataset['description']}")
        print()


def download_dataset(dataset_key, output_dir=None, no_extract=False, organize=False, no_organize=False):
    """Download a specific dataset with options for extraction and organization"""
    if dataset_key not in DATASETS:
        print(f"Error: Dataset '{dataset_key}' not found.")
        list_datasets()
        return False
    
    dataset = DATASETS[dataset_key]
    download_dir = output_dir or dataset['default_dir']
    
    print(f"\nDownloading {dataset['name']}...")
    print(f"Description: {dataset['description']}")
    print(f"Output directory: {download_dir}")
    
    # Create directory if it doesn't exist
    os.makedirs(download_dir, exist_ok=True)
    
    # Determine organization behavior
    should_organize = False
    if organize:
        should_organize = True
    elif not no_organize and dataset.get('organize', False):
        should_organize = True
    
    try:
        # Download files
        if no_extract:
            # Download without extraction
            for url in dataset['urls']:
                filename = url.split('/')[-1]
                filepath = os.path.join(download_dir, filename)
                print(f"Downloading {filename}...")
                download_file(url, filepath)
        else:
            # Download and extract (existing behavior)
            downloadurls(download_dir, dataset['urls'])
            
            # Organize files if needed (KITTI-specific for now)
            if should_organize and dataset_key == 'kitti':
                organize_kitti_dataset(download_dir, dataset)
        
        print(f"\nDownload completed successfully!")
        print(f"Files saved to: {download_dir}")
        return True
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False


def organize_kitti_dataset(download_dir, dataset):
    """Organize KITTI dataset into structured folders"""
    print("Organizing KITTI dataset...")
    
    # Create organized structure
    training_dir = os.path.join(download_dir, 'training')
    os.makedirs(training_dir, exist_ok=True)
    
    # Map extracted folders to organized structure
    folder_mapping = {
        'image_2': os.path.join(training_dir, 'image_2'),
        'label_2': os.path.join(training_dir, 'label_2'), 
        'calib': os.path.join(training_dir, 'calib'),
        'velodyne': os.path.join(training_dir, 'velodyne')
    }
    
    # Move extracted folders to organized structure
    for src_folder, dest_folder in folder_mapping.items():
        src_path = os.path.join(download_dir, src_folder)
        if os.path.exists(src_path):
            if os.path.exists(dest_folder):
                # Remove existing destination
                import shutil
                shutil.rmtree(dest_folder)
            os.rename(src_path, dest_folder)
            print(f"  Organized {src_folder} -> training/{src_folder}")
    
    print("KITTI dataset organization completed!")


def main():
    parser = argparse.ArgumentParser(
        description='Download popular computer vision datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python downloaddata.py --list                    # List all available datasets
  python downloaddata.py --dataset coco128         # Download COCO128 sample dataset
  python downloaddata.py --dataset coco --dir /my/path  # Download COCO to custom directory
  python downloaddata.py --dataset kitti --organize   # Download and organize KITTI dataset
  python downloaddata.py --dataset coco --no-extract  # Download COCO files without extracting
        """
    )
    
    parser.add_argument('--dataset', '-d', 
                       choices=list(DATASETS.keys()),
                       help='Dataset to download')
    
    parser.add_argument('--dir', '-o',
                       help='Output directory (default: dataset-specific directory)')
    
    parser.add_argument('--list', '-l',
                       action='store_true',
                       help='List all available datasets')
    
    parser.add_argument('--no-extract',
                       action='store_true',
                       help='Download files without extracting them')
    
    parser.add_argument('--organize',
                       action='store_true',
                       help='Organize extracted files into structured folders (default for datasets that support it)')
    
    parser.add_argument('--no-organize',
                       action='store_true',
                       help='Skip organizing files into structured folders')
    
    args = parser.parse_args()
    
    if args.list:
        list_datasets()
        return
    
    if not args.dataset:
        print("Error: Please specify a dataset to download or use --list to see available options.")
        parser.print_help()
        return
    
    success = download_dataset(args.dataset, args.dir, args.no_extract, args.organize, args.no_organize)
    if not success:
        exit(1)


if __name__ == '__main__':
    main()