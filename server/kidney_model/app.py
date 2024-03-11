import os
from flask import Flask, render_template, request

import numpy as np
import cv2
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet  # Corrected import statement

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class EfficientNetModel(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetModel, self).__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b7')
        self.fc = nn.Linear(1000, num_classes)  # Adjust the number of output classes

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

# Create an instance of the model
model = EfficientNetModel(num_classes=2)  # 2 classes: benign and malignant

def import_images(path):
    images = []
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = cv2.resize(img, (224, 224))
        images.append(img)
    return images
def func(path):
  images = np.array(import_images(path))

  images = np.array(images, dtype='float') / 255.0

  image = torch.from_numpy(images).unsqueeze(0).float()

  class CustomDataset(Dataset):
      def __init__(self, images, transform=None):
          self.images = images
          self.transform = transform

      def __len__(self):
          return len(self.images)

      def __getitem__(self, idx):
          image = self.images[idx]

          image = image.astype(np.uint8)

          if self.transform:
              image = self.transform(image)

          return image
  # Define transformations for data augmentation and normalization
  test_transform = transforms.Compose([
      transforms.ToPILImage(),
      transforms.ToTensor(),
  ])

  # Create custom datasets
  test_dataset = CustomDataset(images, transform=test_transform)

  # Create dataloaders
  batch_size = 1
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

  y_pred = []
  model = torch.load('server/kidney_model/accuracy_model.pt', map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)
  model.eval()
  with torch.no_grad():
      for images in test_loader:
          images = images.to(device)
          outputs = model(images)
          _, predicted = torch.max(outputs, 1)
          y_pred.extend(predicted.cpu().numpy())

  return y_pred[0]

app = Flask(__name__)

# Define the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        good_morning = func('uploads/'+file.filename)
        result = "No signs of any kidney stone" if good_morning == 0 else "Kidney Stone Detected!"
        return render_template('result.html', filename=filename, result=result)

if __name__ == '__main__':
    app.run(debug=True)
