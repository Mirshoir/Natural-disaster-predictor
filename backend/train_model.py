import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from PIL import UnidentifiedImageError, Image
import os
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample = self.loader(path).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            print(f"Warning: Skipping corrupted image: {path} ({e})")
            return self.__getitem__((index + 1) % len(self.samples))
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

def get_subset_indices_by_class(dataset, max_per_class=50):
    class_counts = defaultdict(int)
    selected_indices = []

    for idx, (path, class_idx) in enumerate(dataset.samples):
        if class_counts[class_idx] < max_per_class:
            selected_indices.append(idx)
            class_counts[class_idx] += 1
        # Stop early if all classes reached max_per_class
        if len(class_counts) == len(dataset.classes) and all(c >= max_per_class for c in class_counts.values()):
            break

    print(f"Selected {len(selected_indices)} images across {len(dataset.classes)} classes (max {max_per_class} per class).")
    return selected_indices

def train_model(data_dir, save_path="backend/model/model.pt", num_epochs=10, batch_size=8):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    full_dataset = SafeImageFolder(data_dir, transform=transform)
    print(f"Found classes: {full_dataset.classes}")

    subset_indices = get_subset_indices_by_class(full_dataset, max_per_class=50)
    subset_dataset = Subset(full_dataset, subset_indices)

    dataloader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=True)

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(full_dataset.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(dataloader):
            try:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if (batch_idx + 1) % 5 == 0:
                    print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")
            except Exception as e:
                print(f"⚠️ Skipping batch {batch_idx + 1} due to error: {e}")

        avg_loss = running_loss / len(dataloader)
        print(f"✅ Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved at: {save_path}")

if __name__ == "__main__":
    train_model(r"C:\Users\dtfygu876\Desktop\natural_disaster_predictor\dataset")
