# backend/model/download_dataset.py
import kagglehub

def download_disaster_dataset():
    path = kagglehub.dataset_download("varpit94/disaster-images-disaster-images-dataset")
    print("✅ Dataset downloaded to:", path)
    return path

if __name__ == "__main__":
    download_disaster_dataset()
