import os
import requests
import zipfile
from pathlib import Path
import gdown


def download_from_google_drive(file_id, destination):
    gdown.download(
        f"https://drive.google.com/uc?id={file_id}", str(destination), quiet=False
    )


def unzip_file(zip_path, extract_to):
    """Extract a zip file to a specified directory."""
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)


if __name__ == "__main__":
    # Replace with your Google Drive file ID
    file_id = "1D7earYq7AlQMWf9MC3s2C37_Uc4VjcXu"

    # Create directories
    download_dir = Path("downloads")
    extract_dir = Path("flowersSquared")

    download_dir.mkdir(exist_ok=True)
    extract_dir.mkdir(exist_ok=True)

    # Download and extract
    zip_path = download_dir / "images.zip"

    print("Downloading file...")
    download_from_google_drive(file_id, zip_path)

    print("Extracting archive...")
    unzip_file(zip_path, extract_dir)

    print("Done!")
