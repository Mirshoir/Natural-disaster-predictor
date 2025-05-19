import os
import shutil

# Source and target base directories
source_base = r"C:\Users\dtfygu876\Desktop\natural_disaster_predictor\disaster-images-dataset"
target_base = r"C:\Users\dtfygu876\Desktop\natural_disaster_predictor\dataset"

# Valid image extensions
valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')

# Recursively walk through the source folder
for root, dirs, files in os.walk(source_base):
    # Only process if this folder contains images
    image_files = [f for f in files if f.lower().endswith(valid_exts)]
    if not image_files:
        continue

    # Use the last folder name as the class label
    class_name = os.path.basename(os.path.normpath(root))
    class_folder = os.path.join(target_base, class_name)

    os.makedirs(class_folder, exist_ok=True)

    for file in image_files:
        src_path = os.path.join(root, file)
        dst_path = os.path.join(class_folder, file)
        try:
            shutil.copyfile(src_path, dst_path)
        except Exception as e:
            print(f"Failed to copy {file}: {e}")

print("✅ Dataset organized successfully!")
