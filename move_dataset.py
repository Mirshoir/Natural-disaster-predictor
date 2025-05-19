import os
import shutil

src_folder = r"C:\Users\dtfygu876\Desktop\natural_disaster_predictor\disaster-images-dataset\Damaged_Infrastructure\Earthquake"
dst_folder = r"C:\Users\dtfygu876\Desktop\natural_disaster_predictor\dataset\Earthquake"

os.makedirs(dst_folder, exist_ok=True)

for file in os.listdir(src_folder):
    if file.lower().endswith(('.jpg', '.jpeg', '.png')):  # Only image files
        src_path = os.path.join(src_folder, file)
        dst_path = os.path.join(dst_folder, file)
        shutil.copyfile(src_path, dst_path)

print("Earthquake images copied to training folder.")
