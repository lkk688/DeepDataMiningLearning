import os
import json
from google.cloud import storage

class WaymoDatasetDownloader:
    def __init__(self, bucket_name='waymo_open_dataset_v_2_0_1', 
                 destination_directory='waymo_dataset'):
        """
        Initialize the Waymo Dataset Downloader.
        
        Args:
            bucket_name (str): Name of the Google Cloud Storage bucket
            destination_directory (str): Local directory to save downloaded files
        """
        self.bucket_name = bucket_name
        self.destination_directory = os.path.abspath(destination_directory)
        self.progress_file = os.path.join(self.destination_directory, 'download_progress.json')
        
        # Ensure the destination directory exists
        os.makedirs(self.destination_directory, exist_ok=True)
        
        # Initialize Google Cloud Storage client
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(bucket_name)
        
        # Load or initialize download progress
        self.download_progress = self._load_progress()

    def _load_progress(self):
        """
        Load download progress from a JSON file or create a new progress tracker.
        
        Returns:
            dict: A dictionary tracking downloaded files
        """
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_progress(self):
        """
        Save download progress to a JSON file.
        """
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(self.download_progress, f, indent=2)
        except IOError as e:
            print(f"Error saving progress: {e}")

    def download_folders(self, folders=None, force_redownload=False):
        """
        Download specific folders from the Waymo dataset.
        
        Args:
            folders (list): List of folder names to download. 
                            If None, lists all available folders.
            force_redownload (bool): If True, redownload all files even if already downloaded
        """
        # List all blobs in the bucket
        blobs = list(self.bucket.list_blobs())
        
        # If no folders specified, list available folders
        if folders is None:
            available_folders = set(blob.name.split('/')[0] for blob in blobs)
            print("Available folders:")
            for folder in sorted(available_folders):
                print(folder)
            return
        
        # Download specified folders
        for folder in folders:
            print(f"\nProcessing folder: {folder}")
            folder_blobs = [blob for blob in blobs if blob.name.startswith(f"{folder}/")]
            
            if not folder_blobs:
                print(f"No blobs found in folder: {folder}")
                continue
            
            for blob in folder_blobs:
                self._download_blob(blob, force_redownload)
        
        # Save final progress
        self._save_progress()
        print("\nDownload completed!")

    def _download_blob(self, blob, force_redownload=False):
        """
        Download a specific blob (file) with resume and progress tracking.
        
        Args:
            blob (Blob): Google Cloud Storage blob to download
            force_redownload (bool): If True, redownload the file
        """
        # Create local file path
        local_file_path = os.path.join(self.destination_directory, blob.name)
        
        # Ensure the local directory exists
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        
        # Check if file should be downloaded
        if not force_redownload and self.download_progress.get(blob.name) == blob.size:
            print(f"Skipping {blob.name} - already downloaded")
            return
        
        # Download the blob
        try:
            print(f"Downloading: {blob.name}")
            blob.download_to_filename(local_file_path)
            
            # Update progress
            self.download_progress[blob.name] = blob.size
            
            print(f"Successfully downloaded: {blob.name}")
        except Exception as e:
            print(f"Error downloading {blob.name}: {e}")

def main():
    # Initialize the downloader
    downloader = WaymoDatasetDownloader(
        bucket_name='waymo_open_dataset_v_2_0_1',
        destination_directory='waymo_dataset'
    )
    
    # Example usage:
    # 1. List available folders
    downloader.download_folders()
    
    # 2. Download specific folders
    downloader.download_folders(['validation'])
    # downloader.download_folders(['training', 'validation'])
    
    # 3. Force redownload of specific folders
    # downloader.download_folders(['training'], force_redownload=True)

if __name__ == '__main__':
    main()

# Prerequisites:
# 1. Install Google Cloud Storage library:
# pip install google-cloud-storage
#
# 2. Set up Google Cloud Authentication:
# - Create a service account in Google Cloud Console
# - Download the JSON key file
# - Set the GOOGLE_APPLICATION_CREDENTIALS environment variable
#   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"