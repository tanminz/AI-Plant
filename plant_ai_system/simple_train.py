"""
Simple training script cho Plant AI System
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import numpy as np
from PIL import Image
from datetime import datetime
import random


class SimplePlantDataset(Dataset):
    """Simple dataset cho training"""
    
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform or self._get_default_transform()
        self.samples = []
        self.classes = []
        
        # Load samples
        self._load_samples()
    
    def _get_default_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
    
    def _load_samples(self):
        """Load samples từ dataset"""
        class_dirs = [d for d in os.listdir(self.data_path) 
                     if os.path.isdir(os.path.join(self.data_path, d))]
        
        self.classes = sorted(class_dirs)
        class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        for class_name in self.classes:
            class_path = os.path.join(self.data_path, class_name)
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_path, img_name)
                    self.samples.append((img_path, class_to_idx[class_name]))
        
        print(f"Loaded {len(self.samples)} samples from {len(self.classes)} classes")
        print(f"Classes: {self.classes}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # Tạo ảnh mẫu nếu không load được
            image = Image.new('RGB', (224, 224), color='green')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def train_simple_model(dataset_path, num_epochs=10):
    """Training model đơn giản"""
    print("Simple Plant AI Training")
    print("=" * 40)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataset
    print(f"Loading dataset from: {dataset_path}")
    dataset = SimplePlantDataset(dataset_path)
    
    # Split dataset
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Model
    num_classes = len(dataset.classes)
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    
    # Optimizer và loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)
        
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Metrics
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs('models', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'classes': dataset.classes
            }, 'models/best_plant_model.pth')
            print(f"New best model saved! Val Acc: {val_acc:.2f}%")
    
    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")
    return best_val_acc


def main():
    """Main function"""
    # Dataset path
    dataset_path = "data/health_monitoring/plant_leaf_diseases"
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at: {dataset_path}")
        print("Please make sure the dataset is properly integrated.")
        return
    
    # Start training
    try:
        best_acc = train_simple_model(dataset_path, num_epochs=5)
        print(f"\n[SUCCESS] Training completed successfully!")
        print(f"Best accuracy: {best_acc:.2f}%")
        print(f"Model saved to: models/best_plant_model.pth")
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
