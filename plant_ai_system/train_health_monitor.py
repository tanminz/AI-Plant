"""
Training script cho Plant Health Monitor
Training ResNet50 model để phân loại bệnh lá cây
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class HealthMonitorTrainer:
    """
    Trainer cho Plant Health Monitor - ResNet50 Classification
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes: List[str] = []
        
        print(f"Health Monitor Trainer initialized on {self.device}")
    
    def train_classification_model(self, train_loader, val_loader, num_classes):
        """
        Training ResNet50 model cho disease classification
        """
        print("Training ResNet50 disease classification model...")
        
        # Khởi tạo model (ResNet50)
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = model.to(self.device)
        
        # Optimizer và scheduler
        optimizer = optim.AdamW(model.parameters(), lr=self.config.get('learning_rate', 0.001), weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.get('epochs', 100))
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0
        patience_counter = 0
        max_epochs = self.config.get('epochs', 100)
        early_stopping_patience = self.config.get('early_stopping_patience', 20)
        
        for epoch in range(max_epochs):
            # Training
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} - Training"):
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
                for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{max_epochs} - Validation"):
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
                # Tạo thư mục models nếu chưa có
                os.makedirs('models', exist_ok=True)
                # Lưu model với tên khớp với app.py
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'classes': self.classes
                }, 'models/best_plant_model.pth')
                print(f"New best model saved! Val Acc: {val_acc:.2f}%")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            scheduler.step()
        
        print(f"Classification training completed! Best accuracy: {best_val_acc:.2f}%")
        return best_val_acc
    
    def _create_classification_loaders(self, dataset_path: Path):
        train_dir = dataset_path / "train"
        val_dir = dataset_path / "val"
        test_dir = dataset_path / "test"
        
        if not train_dir.exists():
            raise FileNotFoundError(f"Train directory not found: {train_dir}")
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        eval_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        val_dataset = datasets.ImageFolder(val_dir, transform=eval_transform)
        test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)
        
        self.classes = train_dataset.classes
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader, len(self.classes)
    
    def _evaluate(self, model: nn.Module, data_loader: DataLoader) -> Dict:
        model.eval()
        preds = []
        labels = []
        
        with torch.no_grad():
            for images, targets in data_loader:
                images = images.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                preds.extend(predicted.cpu().numpy())
                labels.extend(targets.numpy())
        
        report = classification_report(labels, preds, target_names=self.classes, output_dict=True)
        matrix = confusion_matrix(labels, preds).tolist()
        accuracy = report['accuracy']
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': matrix
        }
    
    def train(self, dataset_path):
        """
        Training chính cho Health Monitor - ResNet50 Classification
        """
        print("Starting Plant Health Monitor training...")
        print("=" * 50)
        dataset_path = Path(dataset_path)
        
        # Training classification model
        print("\nTraining ResNet50 disease classification model...")
        train_loader, val_loader, test_loader, num_classes = self._create_classification_loaders(dataset_path)
        
        classification_acc = self.train_classification_model(
            train_loader, val_loader, num_classes
        )
        
        # Đánh giá trên tập test
        evaluation_metrics = {}
        if classification_acc > 0:
            best_model_path = Path('models/best_plant_model.pth')
            if best_model_path.exists():
                print("\nEvaluating on test set...")
                checkpoint = torch.load(best_model_path, map_location=self.device)
                model = models.resnet50(pretrained=False)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
                model.load_state_dict(checkpoint['model_state_dict'])
                model = model.to(self.device)
                evaluation_metrics = self._evaluate(model, test_loader)
                print(f"Test Accuracy: {evaluation_metrics['accuracy']*100:.2f}%")
        
        # Tạo training report
        training_report = {
            'timestamp': datetime.now().isoformat(),
            'training': 'completed' if classification_acc > 0 else 'failed',
            'best_validation_accuracy': classification_acc,
            'test_accuracy': evaluation_metrics.get('accuracy', 0) * 100 if evaluation_metrics else 0,
            'classes': self.classes,
            'num_classes': num_classes,
            'config': self.config,
            'evaluation': evaluation_metrics
        }
        
        # Lưu training report
        os.makedirs('models', exist_ok=True)
        with open('models/training_report.json', 'w') as f:
            json.dump(training_report, f, indent=2)
        
        print("\n" + "=" * 50)
        print("Training completed!")
        print(f"Best Validation Accuracy: {classification_acc:.2f}%")
        if evaluation_metrics:
            print(f"Test Accuracy: {evaluation_metrics.get('accuracy', 0)*100:.2f}%")
        print(f"Model saved: models/best_plant_model.pth")
        print(f"Report saved: models/training_report.json")
        
        return training_report


def main():
    """Main training function"""
    print("Plant Health Monitor Training - ResNet50")
    print("=" * 50)
    
    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Training config
    training_config = {
        'batch_size': config.get('training', {}).get('batch_size', 32),
        'learning_rate': config.get('training', {}).get('learning_rate', 0.001),
        'epochs': config.get('training', {}).get('epochs', 100),
        'early_stopping_patience': config.get('training', {}).get('early_stopping_patience', 20)
    }
    
    # Lấy đường dẫn dataset từ config
    dataset_path = Path(config['data_paths']['plant_leaf_diseases'])
    
    # Kiểm tra dataset có tồn tại không
    if not dataset_path.exists():
        print(f"ERROR: Dataset path not found: {dataset_path}")
        print("Please check config.json and ensure dataset is available.")
        return
    
    # Initialize trainer
    trainer = HealthMonitorTrainer(training_config)
    
    # Start training
    try:
        report = trainer.train(dataset_path)
        print("\nTraining completed successfully!")
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
