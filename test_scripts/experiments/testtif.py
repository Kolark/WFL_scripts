from PIL import Image
import PIL.ExifTags

# Open the image
with Image.open("/home/felipe/Desktop/Trabajo/Data/COL_wind-speed_100m.tif") as img:
    # Get basic image attributes
    width, height = img.size
    mode = img.mode  # e.g., 'RGB', 'L', 'I;16'
    num_frames = getattr(img, "n_frames", 1)
    
    print(f"Dimensions: {width} x {height} pixels")
    print(f"Color Mode: {mode}")
    print(f"Number of Pages/Frames: {num_frames}")

    # Read TIFF/EXIF metadata tag dictionary
    meta_data = img.tag_v2
    print("\n--- TIFF Metadata Tags ---")
    for tag_id, value in meta_data.items():
        tag_name = PIL.ExifTags.TAGS.get(tag_id, tag_id)
        print(f"{tag_name} ({tag_id}): {value}")