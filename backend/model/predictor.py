import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
from torchvision import models


class Predictor:
    def __init__(self, model_path: str, class_names: list):
        """
        Initialize the predictor with the given model path and class names.
        """
        self.class_names = class_names

        # Load a ResNet50 model and replace the classifier head
        self.model = models.resnet50(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, len(class_names))

        # Load the trained model weights
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        self.model.eval()  # Set the model to evaluation mode

        # Define image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_base64: str) -> str:
        """
        Predict the class of a base64-encoded image.

        Args:
            image_base64 (str): Base64-encoded image string.

        Returns:
            str: Predicted class name.
        """
        # Decode and preprocess the image
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image = self.transform(image).unsqueeze(0)  # Add batch dimension

        # Perform prediction
        with torch.no_grad():
            outputs = self.model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_index = torch.argmax(probabilities, dim=1).item()

        # Return the predicted class name
        return self.class_names[predicted_index]
