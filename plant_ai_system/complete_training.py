"""
Complete Training Script - Tích hợp tất cả datasets để nhận diện đầy đủ
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import json
import random
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
import shutil
from collections import defaultdict


class CompleteDatasetLoader(Dataset):
    """Complete dataset loader cho tất cả datasets"""
    
    def __init__(self, data_config, transform=None, mode='train'):
        self.data_config = data_config
        self.mode = mode
        self.transform = transform or self._get_default_transform()
        self.samples = []
        self.classes = []
        self.class_mapping = {}
        
        # Load samples từ tất cả datasets
        self._load_all_datasets()
    
    def _get_default_transform(self):
        if self.mode == 'train':
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225])
            ])
    
    def _load_all_datasets(self):
        """Load samples từ tất cả datasets"""
        all_classes = set()
        dataset_stats = {}
        
        for dataset_name, dataset_info in self.data_config.items():
            print(f"\nLoading dataset: {dataset_name}")
            print(f"Path: {dataset_info['path']}")
            
            if not os.path.exists(dataset_info['path']):
                print(f"Dataset not found: {dataset_info['path']}")
                continue
            
            dataset_samples = 0
            
            # Load classes từ dataset
            if dataset_info['type'] == 'structured':
                # Dataset có cấu trúc train/val/test
                mode_path = os.path.join(dataset_info['path'], self.mode)
                if os.path.exists(mode_path):
                    classes = [d for d in os.listdir(mode_path) 
                              if os.path.isdir(os.path.join(mode_path, d))]
                else:
                    # Fallback to root directory or train folder
                    if os.path.exists(os.path.join(dataset_info['path'], 'train')):
                        # Use train folder if mode folder doesn't exist
                        mode_path = os.path.join(dataset_info['path'], 'train')
                        classes = [d for d in os.listdir(mode_path) 
                                  if os.path.isdir(os.path.join(mode_path, d))]
                    else:
                        # Fallback to root directory
                        classes = [d for d in os.listdir(dataset_info['path']) 
                                  if os.path.isdir(os.path.join(dataset_info['path'], d))]
                        mode_path = dataset_info['path']
            else:
                # Dataset có cấu trúc flat
                # Check if there's a train subfolder
                train_subfolder = os.path.join(dataset_info['path'], 'train')
                if os.path.exists(train_subfolder) and self.mode == 'train':
                    mode_path = train_subfolder
                    classes = [d for d in os.listdir(mode_path) 
                              if os.path.isdir(os.path.join(mode_path, d))]
                else:
                    # Use root directory
                    classes = [d for d in os.listdir(dataset_info['path']) 
                              if os.path.isdir(os.path.join(dataset_info['path'], d))]
                    mode_path = dataset_info['path']
            
            for class_name in classes:
                if class_name in ['splits', '__pycache__', '.git']:
                    continue
                    
                class_path = os.path.join(mode_path, class_name)
                if not os.path.isdir(class_path):
                    continue
                
                # Normalize class name
                normalized_class = self._normalize_class_name(class_name, dataset_name)
                all_classes.add(normalized_class)
                
                # Load images trong class
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        img_path = os.path.join(class_path, img_name)
                        self.samples.append({
                            'image_path': img_path,
                            'class': normalized_class,
                            'original_class': class_name,
                            'dataset': dataset_name,
                            'category': dataset_info.get('category', 'unknown')
                        })
                        dataset_samples += 1
            
            dataset_stats[dataset_name] = dataset_samples
            print(f"  Loaded {dataset_samples} samples")
        
        self.classes = sorted(list(all_classes))
        self.class_mapping = {cls: idx for idx, cls in enumerate(self.classes)}
        
        print(f"\n=== COMPLETE DATASET SUMMARY ===")
        print(f"Total classes: {len(self.classes)}")
        print(f"Total samples: {len(self.samples)}")
        print(f"Mode: {self.mode}")
        
        for dataset_name, count in dataset_stats.items():
            print(f"{dataset_name}: {count} samples")
        
        # Class distribution
        class_counts = defaultdict(int)
        for sample in self.samples:
            class_counts[sample['class']] += 1
        
        print(f"\nClass distribution:")
        for cls in sorted(class_counts.keys()):
            print(f"  {cls}: {class_counts[cls]} samples")
    
    def _normalize_class_name(self, class_name, dataset_name):
        """Normalize class names across datasets"""
        # Remove dataset prefixes and normalize
        class_name = class_name.replace('___', '_').replace('__', '_')
        
        # Map similar classes
        if 'healthy' in class_name.lower():
            if 'durian' in dataset_name.lower():
                return 'Durian_Healthy'
            elif 'apple' in class_name.lower():
                return 'Apple_Healthy'
            elif 'cherry' in class_name.lower():
                return 'Cherry_Healthy'
            elif 'blueberry' in class_name.lower():
                return 'Blueberry_Healthy'
            elif 'cashew' in dataset_name.lower():
                return 'Cashew_Healthy'
            elif 'cassava' in dataset_name.lower():
                return 'Cassava_Healthy'
            elif 'maize' in dataset_name.lower():
                return 'Maize_Healthy'
            elif 'tomato' in dataset_name.lower():
                return 'Tomato_Healthy'
            else:
                return 'Healthy'
        elif 'scab' in class_name.lower():
            return 'Apple_Scab'
        elif 'black_rot' in class_name.lower():
            return 'Apple_Black_Rot'
        elif 'rust' in class_name.lower():
            if 'cedar' in class_name.lower():
                return 'Apple_Cedar_Rust'
            elif 'common' in class_name.lower() or 'corn' in class_name.lower():
                return 'Corn_Common_Rust'
            else:
                return 'Cashew_Red_Rust'
        elif 'powdery_mildew' in class_name.lower():
            return 'Cherry_Powdery_Mildew'
        elif 'algal' in class_name.lower():
            return 'Durian_Algal_Disease'
        elif 'blight' in class_name.lower():
            if 'durian' in dataset_name.lower():
                return 'Durian_Blight'
            elif 'tomato' in dataset_name.lower():
                return 'Tomato_Leaf_Blight'
            elif 'maize' in dataset_name.lower():
                return 'Maize_Leaf_Blight'
            else:
                return 'Blight'
        elif 'colletotrichum' in class_name.lower():
            return 'Durian_Anthracnose'
        elif 'anthracnose' in class_name.lower():
            if 'cashew' in dataset_name.lower():
                return 'Cashew_Anthracnose'
            else:
                return 'Anthracnose'
        elif 'phomopsis' in class_name.lower():
            return 'Durian_Phomopsis'
        elif 'rhizoctonia' in class_name.lower():
            return 'Durian_Rhizoctonia'
        elif 'cercospora' in class_name.lower() or 'gray_leaf_spot' in class_name.lower() or 'gray leaf spot' in class_name.lower():
            return 'Corn_Cercospora_Leaf_Spot'
        elif 'brown_spot' in class_name.lower():
            return 'Cassava_Brown_Spot'
        elif 'bacterial_blight' in class_name.lower():
            return 'Cassava_Bacterial_Blight'
        elif 'mosaic' in class_name.lower():
            return 'Cassava_Mosaic'
        elif 'green_mite' in class_name.lower():
            return 'Cassava_Green_Mite'
        elif 'leaf_miner' in class_name.lower():
            return 'Cashew_Leaf_Miner'
        elif 'gumosis' in class_name.lower():
            return 'Cashew_Gumosis'
        elif 'fall_armyworm' in class_name.lower():
            return 'Maize_Fall_Armyworm'
        elif 'grasshoper' in class_name.lower():
            return 'Maize_Grasshopper'
        elif 'leaf_beetle' in class_name.lower():
            return 'Maize_Leaf_Beetle'
        elif 'leaf_spot' in class_name.lower():
            if 'maize' in dataset_name.lower():
                return 'Maize_Leaf_Spot'
            elif 'tomato' in dataset_name.lower():
                return 'Tomato_Septoria_Leaf_Spot'
            else:
                return 'Leaf_Spot'
        elif 'streak_virus' in class_name.lower():
            return 'Maize_Streak_Virus'
        elif 'leaf_curl' in class_name.lower():
            return 'Tomato_Leaf_Curl'
        elif 'verticillium_wilt' in class_name.lower():
            return 'Tomato_Verticillium_Wilt'
        elif 'background' in class_name.lower():
            return 'Background'
        elif 'corn' in class_name.lower() and 'healthy' in class_name.lower():
            return 'Corn_Healthy'
        else:
            # Keep original class name if no specific mapping found
            normalized = class_name.replace('___', '_').replace('__', '_').strip('_')
            return normalized
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            image = Image.open(sample['image_path']).convert('RGB')
        except Exception as e:
            # Tạo ảnh mẫu nếu không load được
            print(f"Error loading image {sample['image_path']}: {e}")
            image = Image.new('RGB', (224, 224), color='green')
        
        if self.transform:
            image = self.transform(image)
        
        # Convert class name to index
        class_idx = self.class_mapping[sample['class']]
        
        return image, class_idx
    
    def get_class_name(self, class_idx):
        """Get class name from index"""
        return self.classes[class_idx] if class_idx < len(self.classes) else "Unknown"


class CompletePlantTrainer:
    """Complete trainer cho tất cả datasets"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.classes = None
        
        print(f"Complete Plant Trainer initialized on {self.device}")
    
    def load_datasets(self):
        """Load tất cả datasets"""
        data_config = {
            'plant_leaf_diseases': {
                'path': 'data/health_monitoring/plant_leaf_diseases',
                'type': 'flat',
                'category': 'general_plant_diseases'
            },
            'durian_diseases': {
                'path': 'data/A Durian Leaf Image Dataset/A Durian Leaf Image Dataset of Common Diseases in Vietnam for Agricultural Diagnosis/Durian_Leaf_Diseases',
                'type': 'structured',
                'category': 'durian_diseases'
            },
            'crop_pest_diseases': {
                'path': 'data/Crop_Pest_Disease_Detection/Dataset for Crop Pest and Disease Detection/CCMT Dataset-Augmented',
                'type': 'structured',
                'category': 'crop_diseases'
            },
            'plant_diseases_9layer': {
                'path': 'data/Data for Identification of Plant Leaf Diseases Using a 9-layer Deep Convolutional Neural Network/extracted_without_augmentation/Plant_leave_diseases_dataset_without_augmentation',
                'type': 'flat',
                'category': 'general_plant_diseases'
            },
            'new_plant_diseases': {
                'path': 'data/archive/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train',
                'type': 'flat',
                'category': 'general_plant_diseases'
            }
        }
        
        # Tạo dataset loaders
        train_dataset = CompleteDatasetLoader(data_config, mode='train')
        val_dataset = CompleteDatasetLoader(data_config, mode='val')
        
        self.classes = train_dataset.classes
        
        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
        
        print(f"\nTrain samples: {len(train_dataset)}")
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
        
        print(f"Starting complete training for {epochs} epochs...")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
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
                
                if batch_idx % 100 == 0:
                    print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
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
                self.save_model(f"complete_plant_model_epoch_{epoch+1}.pth", epoch, val_loss)
                # Also save as best_plant_model.pth for web app
                self.save_model("best_plant_model.pth", epoch, val_loss)
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
            'timestamp': datetime.now().isoformat(),
            'total_epochs': epochs
        }
        
        with open('models/complete_training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"\nComplete training completed! Best validation accuracy: {best_val_acc:.2f}%")
        return best_val_acc
    
    def save_model(self, filename, epoch, loss):
        """Save model checkpoint"""
        os.makedirs('models', exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'classes': self.classes,
            'loss': loss,
            'num_classes': len(self.classes),
            'model_type': 'complete_plant_ai'
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
    """Main complete training function"""
    print("COMPLETE PLANT AI TRAINING - All Datasets")
    print("=" * 60)
    
    # Training config
    config = {
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 30,  # Increased for comprehensive training
        'weight_decay': 0.01
    }
    
    # Initialize trainer
    trainer = CompletePlantTrainer(config)
    
    # Load datasets
    print("\n1. Loading all datasets...")
    train_loader, val_loader = trainer.load_datasets()
    
    # Create model
    print("\n2. Creating complete model...")
    trainer.create_model()
    
    # Train model
    print("\n3. Starting complete training...")
    best_acc = trainer.train_model(train_loader, val_loader, epochs=config['epochs'])
    
    # Test model
    print("\n4. Testing complete model...")
    test_acc = trainer.test_model(val_loader)
    
    print(f"\n[SUCCESS] COMPLETE TRAINING COMPLETED!")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Test accuracy: {test_acc:.2f}%")
    print(f"Total classes: {len(trainer.classes)}")
    print(f"Model saved to: models/complete_plant_model_epoch_*.pth")
    print(f"History saved to: models/complete_training_history.json")


if __name__ == "__main__":
    main()

