import os
import cv2
from tqdm import tqdm

# First, collect all jpg files to get total count
jpg_files = []
for root, dirs, files in os.walk("102flowers/jpg"):
    for file in files:
        if file.lower().endswith(".jpg"):
            jpg_files.append(os.path.join(root, file))

# Process files with progress bar
for file_path in tqdm(jpg_files, desc="Processing images"):
    # Extract filename from path
    file = os.path.basename(file_path)

    # Process the jpg file here      # Read the image
    img = cv2.imread(file_path)

    # Get image dimensions
    height, width = img.shape[:2]

    # Calculate the size of the center square
    size = min(height, width)

    # Calculate crop coordinates for center square
    start_x = (width - size) // 2
    start_y = (height - size) // 2
    end_x = start_x + size
    end_y = start_y + size

    # Crop the center square
    cropped_img = img[start_y:end_y, start_x:end_x]

    # Resize to 512x512 pixels
    resized_img = cv2.resize(cropped_img, (512, 512))

    # Create output directory if it doesn't exist
    output_dir = "flowersSquared"
    os.makedirs(output_dir, exist_ok=True)

    # Save the processed image
    output_path = os.path.join(output_dir, file)
    cv2.imwrite(output_path, resized_img)
