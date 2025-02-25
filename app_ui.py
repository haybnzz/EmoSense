import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the CNN model (must match the architecture used during training)
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):  # Updated to 7 classes
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 1 input channel for grayscale
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 6 * 6, 128)  # 48x48 -> 6x6 after 3 pooling layers
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)  # Number of classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Define the image preprocessing (must match training)
preprocess = transforms.Compose([
    transforms.Grayscale(),  # Convert to grayscale
    transforms.Resize((48, 48)),  # Resize to 48x48
    transforms.ToTensor(),  # Convert to tensor and normalize to [0, 1]
])

# Load the class names (update this based on your training subfolders)
class_names = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']  # 7 classes

# Function to predict emotion from an image
def predict_emotion(image, model_path='emotion_model.pth'):
    # Load the model
    model = EmotionCNN(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set to evaluation mode

    # Preprocess the image
    try:
        image = preprocess(image)
        image = image.unsqueeze(0)  # Add batch dimension (1, 1, 48, 48)
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None, None

    # Perform prediction
    with torch.no_grad():
        image = image.to(device)
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)  # Convert to probabilities
        confidence, predicted = torch.max(probabilities, 1)  # Get highest probability and class
        emotion = class_names[predicted.item()]
        confidence_score = confidence.item() * 100  # Convert to percentage

    return emotion, confidence_score

# Streamlit app
def main():
    st.title("Emotion Detection App")
    st.write("Upload an image to detect the emotion using a pre-trained model.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Predict emotion
        emotion, confidence = predict_emotion(image)
        
        if emotion:
            st.success(f"Predicted Emotion: **{emotion}** (Confidence: {confidence:.2f}%)")
        else:
            st.error("Failed to predict emotion. Please try another image.")

if __name__ == "__main__":
    main()
