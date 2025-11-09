"""
Training script cho Module B - Plant Health Monitor
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
from datetime import datetime
# import wandb  # Optional: install with pip install wandb
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import cv2
from ultralytics import YOLO
# import albumentations as A  # Optional: install with pip install albumentations
# from albumentations.pytorch import ToTensorV2

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from module_b_health_monitor.plant_health_monitor import PlantHealthMonitor
from utils.data_loader import create_dataloader, load_health_monitoring_dataset


class HealthMonitorTrainer:
    """
    Trainer cho Plant Health Monitor
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Khởi tạo YOLO model
        self.yolo_model = YOLO('yolov8n.pt')  # nano version for speed
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        print(f"Health Monitor Trainer initialized on {self.device}")
    
    def prepare_yolo_dataset(self, data_path, output_path):
        """
        Chuẩn bị dataset cho YOLO training
        """
        print("Preparing YOLO dataset...")
        
        # Tạo cấu trúc thư mục YOLO
        os.makedirs(os.path.join(output_path, 'images', 'train'), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'images', 'val'), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'labels', 'train'), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'labels', 'val'), exist_ok=True)
        
        # Load dataset metadata
        metadata_path = os.path.join(data_path, 'dataset_metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Tạo class mapping
        class_mapping = {}
        for class_id, class_info in metadata['class_mapping'].items():
            class_mapping[class_info['name']] = int(class_id)
        
        # Tạo YOLO config
        yolo_config = {
            'path': output_path,
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(class_mapping),
            'names': list(class_mapping.keys())
        }
        
        with open(os.path.join(output_path, 'data.yaml'), 'w') as f:
            import yaml
            yaml.dump(yolo_config, f)
        
        print(f"YOLO dataset prepared in: {output_path}")
        return yolo_config
    
    def train_yolo_model(self, dataset_path, epochs=100):
        """
        Training YOLO model cho disease detection
        """
        print("Training YOLO model for disease detection...")
        
        # Chuẩn bị dataset
        yolo_data_path = os.path.join(dataset_path, 'yolo_dataset')
        yolo_config = self.prepare_yolo_dataset(dataset_path, yolo_data_path)
        
        # Training parameters
        training_params = {
            'data': os.path.join(yolo_data_path, 'data.yaml'),
            'epochs': epochs,
            'imgsz': 640,
            'batch': 16,
            'device': self.device,
            'project': 'models/health_monitor',
            'name': 'yolo_disease_detection',
            'save': True,
            'save_period': 10,
            'patience': 20,
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'pose': 12.0,
            'kobj': 1.0,
            'label_smoothing': 0.0,
            'nbs': 64,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0
        }
        
        try:
            # Start training
            results = self.yolo_model.train(**training_params)
            
            print("YOLO training completed successfully!")
            print(f"Results saved in: models/health_monitor/yolo_disease_detection/")
            
            return results
            
        except Exception as e:
            print(f"YOLO training failed: {e}")
            return None
    
    def train_classification_model(self, train_loader, val_loader, num_classes):
        """
        Training classification model cho disease classification
        """
        print("Training disease classification model...")
        
        # Khởi tạo model (ResNet50)
        import torchvision.models as models
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = model.to(self.device)
        
        # Optimizer và scheduler
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(100):
            # Training
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/100 - Training"):
                images, labels = images.to(self.device), labels.to(self.device)
                
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
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/100 - Validation"):
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            # Calculate metrics
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"         Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss
                }, 'models/health_monitor/best_classification_model.pth')
                print(f"New best model saved! Val Acc: {val_acc:.2f}%")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= 20:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            scheduler.step()
        
        print(f"Classification training completed! Best accuracy: {best_val_acc:.2f}%")
        return best_val_acc
    
    def train(self, dataset_path):
        """
        Training chính cho Health Monitor
        """
        print("Starting Health Monitor training...")
        
        # 1. Training YOLO cho object detection
        print("\n1. Training YOLO for disease detection...")
        yolo_results = self.train_yolo_model(dataset_path)
        
        # 2. Training classification model
        print("\n2. Training disease classification model...")
        
        # Load dataset metadata
        metadata_path = os.path.join(dataset_path, 'dataset_metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        num_classes = metadata['total_classes']
        
        # Tạo dataloader cho classification
        try:
            train_loader, _ = load_health_monitoring_dataset(
                dataset_path, "train", batch_size=32
            )
            val_loader, _ = load_health_monitoring_dataset(
                dataset_path, "val", batch_size=32
            )
            
            classification_acc = self.train_classification_model(
                train_loader, val_loader, num_classes
            )
            
        except Exception as e:
            print(f"Classification training failed: {e}")
            classification_acc = 0
        
        # 3. Tạo training report
        training_report = {
            'timestamp': datetime.now().isoformat(),
            'yolo_training': 'completed' if yolo_results else 'failed',
            'classification_training': 'completed' if classification_acc > 0 else 'failed',
            'best_classification_accuracy': classification_acc,
            'dataset_info': metadata,
            'config': self.config
        }
        
        # Lưu training report
        with open('models/health_monitor/training_report.json', 'w') as f:
            json.dump(training_report, f, indent=2)
        
        print("\nHealth Monitor training completed!")
        print(f"YOLO training: {'Success' if yolo_results else 'Failed'}")
        print(f"Classification accuracy: {classification_acc:.2f}%")
        
        return training_report


def main():
    """Main training function"""
    print("Plant Health Monitor Training")
    print("=" * 50)
    
    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Training config
    training_config = {
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 100,
        'early_stopping_patience': 20
    }
    
    # Dataset path
    dataset_path = "data/health_monitoring/plant_leaf_diseases"
    
    # Initialize trainer
    trainer = HealthMonitorTrainer(training_config)
    
    # Start training
    try:
        report = trainer.train(dataset_path)
        print("Training completed successfully!")
        print(f"Report saved: models/health_monitor/training_report.json")
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
