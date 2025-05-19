import torch
import torch.nn as nn

class MockModel(nn.Module):
    def __init__(self, num_classes=3):
        super(MockModel, self).__init__()
        # A dummy linear layer
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(224 * 224 * 3, num_classes)  # For RGB images 224x224

    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)

if __name__ == "__main__":
    model = MockModel()
    torch.save(model, "backend/model/model.pt")
    print("Mock model saved to backend/model/model.pt")
