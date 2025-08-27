import tarfile
import glob
import os

# Path to where your zip files are stored
zip_dir = r"E:\Shared\Dataset\NuScenes"
# Destination folder
extract_dir = os.path.join(zip_dir, "v1.0-trainval")

# Make sure the destination directory exists
os.makedirs(extract_dir, exist_ok=True)

# Find all blob tgz files
zip_files = glob.glob(os.path.join(zip_dir, "v1.0-trainval*_blobs.tgz"))

print(f"Found {len(zip_files)} tgz files.")

# Extract each file
for zip_path in zip_files:
    print(f"Extracting {zip_path}...")
    with tarfile.open(zip_path, 'r:gz') as tar_ref:
        tar_ref.extractall(extract_dir)

print("Done extracting all blobs.")
