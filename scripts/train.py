import torch
import torch.optim as optim
import torch.nn as nn
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from models.simple_yolo import SimpleYOLOv8
from utils.dataset import get_data_loader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='data/train/', help='data directory')
parser.add_argument('--batch', type=int, default=16, help='batch size')
parser.add_argument('--model', type=str, default='yolov8_simple.pth', help='model path')
parser.add_argument('--epoch', type=int, default=10, help='number of epochs')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleYOLOv8(num_classes=5).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

train_loader = get_data_loader("sample_data/train/", batch_size=4)
val_loader = get_data_loader("sample_data/val/", batch_size=4)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), 'yolov8_simple.pth')