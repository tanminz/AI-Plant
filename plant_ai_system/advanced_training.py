"""
Advanced Training Script với tất cả datasets mới
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms, models
from PIL import Image
import json
import random
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
import shutil


class MultiDatasetLoader(Dataset):
    """Dataset loader cho nhiều datasets"""
    
    def __init__(self, data_paths, transform=None):
        self.data_paths = data_paths
        self.transform = transform or self._get_default_transform()
        self.samples = []
        self.classes = []
        
        # Load samples từ tất cả datasets
        self._load_all_datasets()
    
    def _get_default_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
    
    def _load_all_datasets(self):
        """Load samples từ tất cả datasets"""
        all_classes = set()
        
        for dataset_name, dataset_path in self.data_paths.items():
            print(f"Loading dataset: {dataset_name}")
            print(f"Path: {dataset_path}")
            
            if not os.path.exists(dataset_path):
                print(f"Dataset not found: {dataset_path}")
                continue
            
            # Load classes từ dataset
            classes = [d for d in os.listdir(dataset_path) 
                      if os.path.isdir(os.path.join(dataset_path, d))]
            
            for class_name in classes:
                class_path = os.path.join(dataset_path, class_name)
                all_classes.add(class_name)
                
                # Load images trong class
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_path, img_name)
                        self.samples.append({
                            'image_path': img_path,
                            'class': class_name,
                            'dataset': dataset_name
                        })
        
        self.classes = sorted(list(all_classes))
        print(f"Total classes: {len(self.classes)}")
        print(f"Total samples: {len(self.samples)}")
        print(f"Classes: {self.classes}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            image = Image.open(sample['image_path']).convert('RGB')
        except:
            # Tạo ảnh mẫu nếu không load được
            image = Image.new('RGB', (224, 224), color='green')
        
        if self.transform:
            image = self.transform(image)
        
        # Convert class name to index
        class_idx = self.classes.index(sample['class'])
        
        return image, class_idx
    
    def get_class_name(self, class_idx):
        """Get class name from index"""
        return self.classes[class_idx] if class_idx < len(self.classes) else "Unknown"


class AdvancedPlantTrainer:
    """Advanced trainer cho multiple datasets"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.classes = None
        
        print(f"Advanced Plant Trainer initialized on {self.device}")
    
    def load_datasets(self):
        """Load tất cả datasets"""
        datasets = {
            'plant_leaf_diseases': 'data/health_monitoring/plant_leaf_diseases',
            'durian_diseases': 'data/A Durian Leaf Image Dataset/A Durian Leaf Image Dataset of Common Diseases in Vietnam for Agricultural Diagnosis/Durian_Leaf_Diseases/train',
            'augmented_plant_diseases': 'data/archive/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train'
        }
        
        # Tạo dataset loader
        dataset_loader = MultiDatasetLoader(datasets)
        self.classes = dataset_loader.classes
        
        # Split dataset
        total_size = len(dataset_loader)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset_loader, [train_size, val_size]
        )
        
        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def create_model(self):
        """Tạo model mới với số classes mới"""
        num_classes = len(self.classes)
        
        # Sử dụng ResNet50
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = model.to(self.device)
        
        self.model = model
        print(f"Model created with {num_classes} classes")
        return model
    
    def train_model(self, train_loader, val_loader, epochs=20):
        """Training model"""
        if self.model is None:
            self.create_model()
        
        # Optimizer và scheduler
        optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            # Calculate metrics
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            # Store metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model(f"advanced_plant_model_epoch_{epoch+1}.pth", epoch, val_loss)
                print(f"New best model saved! Val Acc: {val_acc:.2f}%")
            
            scheduler.step()
        
        # Save training history
        history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'classes': self.classes,
            'best_val_acc': best_val_acc,
            'timestamp': datetime.now().isoformat()
        }
        
        with open('models/advanced_training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")
        return best_val_acc
    
    def save_model(self, filename, epoch, loss):
        """Save model checkpoint"""
        os.makedirs('models', exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'classes': self.classes,
            'loss': loss,
            'num_classes': len(self.classes)
        }
        
        torch.save(checkpoint, f'models/{filename}')
    
    def test_model(self, test_loader):
        """Test model performance"""
        if self.model is None:
            print("Model not loaded!")
            return
        
        self.model.eval()
        correct = 0
        total = 0
        class_correct = [0] * len(self.classes)
        class_total = [0] * len(self.classes)
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Per-class accuracy
                for i in range(labels.size(0)):
                    label = labels[i]
                    class_correct[label] += (predicted[i] == label).item()
                    class_total[label] += 1
        
        # Overall accuracy
        accuracy = 100. * correct / total
        print(f"Overall Test Accuracy: {accuracy:.2f}%")
        
        # Per-class accuracy
        print("\nPer-class Accuracy:")
        for i, class_name in enumerate(self.classes):
            if class_total[i] > 0:
                class_acc = 100. * class_correct[i] / class_total[i]
                print(f"  {class_name}: {class_acc:.2f}%")
        
        return accuracy


def main():
    """Main training function"""
    print("Advanced Plant AI Training with Multiple Datasets")
    print("=" * 60)
    
    # Training config
    config = {
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 20,
        'weight_decay': 0.01
    }
    
    # Initialize trainer
    trainer = AdvancedPlantTrainer(config)
    
    # Load datasets
    print("\n1. Loading datasets...")
    train_loader, val_loader = trainer.load_datasets()
    
    # Create model
    print("\n2. Creating model...")
    trainer.create_model()
    
    # Train model
    print("\n3. Training model...")
    best_acc = trainer.train_model(train_loader, val_loader, epochs=20)
    
    # Test model
    print("\n4. Testing model...")
    test_acc = trainer.test_model(val_loader)
    
    print(f"\n[SUCCESS] Advanced training completed!")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Test accuracy: {test_acc:.2f}%")
    print(f"Total classes: {len(trainer.classes)}")
    print(f"Model saved to: models/advanced_plant_model_epoch_*.pth")


if __name__ == "__main__":
    main()







