from pathlib import Path
from PIL import Image

def resize_images_in_folder(folder_path, max_size_mb=1):
    max_size_bytes = max_size_mb * 1024 * 1024  # 1MB in bytes
    folder = Path(folder_path)

    for file_path in folder.glob('*.jpg'):
        with Image.open(file_path) as img:
            # Check if the file size is greater than the max size
            if file_path.stat().st_size > max_size_bytes:
                # Calculate the reduction factor
                reduction_factor = (max_size_bytes / file_path.stat().st_size) ** 0.5
                new_dimensions = (int(img.width * reduction_factor), int(img.height * reduction_factor))
                
                # Resize the image
                img = img.resize(new_dimensions, Image.ANTIALIAS)
                
                # Save the resized image
                img.save(file_path, optimize=True, quality=85)

folder_path = 'floodnet'
resize_images_in_folder(folder_path)