"""
Training script cho Module A - Plant Species Recognition
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
from datetime import datetime
import wandb
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from module_a_species_recognition.plant_species_classifier import PlantSpeciesClassifier
from utils.data_loader import create_dataloader, load_plantclef_dataset


class SpeciesRecognitionTrainer:
    """
    Trainer cho Plant Species Recognition
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Khởi tạo model
        self.model = PlantSpeciesClassifier(
            model_type=config.get('model_type', 'clip'),
            num_classes=config.get('num_classes', 80000)
        )
        
        # Khởi tạo optimizer và scheduler
        self.optimizer = optim.AdamW(
            self.model.model.parameters(),
            lr=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('epochs', 100)
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        print(f"Species Recognition Trainer initialized on {self.device}")
    
    def train_epoch(self, train_loader):
        """Train một epoch"""
        self.model.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.model.model_type == "clip":
                # CLIP training với contrastive learning
                image_features = self.model.model.get_image_features(images)
                # Tạo text features từ labels (cần implement text prompts)
                # Tạm thời sử dụng classification head
                logits = self.model.classification_head(image_features)
            else:
                # CNN/ViT training
                logits = self.model.model(images)
            
            # Calculate loss
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, val_loader):
        """Validate một epoch"""
        self.model.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                if self.model.model_type == "clip":
                    image_features = self.model.model.get_image_features(images)
                    logits = self.model.classification_head(image_features)
                else:
                    logits = self.model.model(images)
                
                # Calculate loss
                loss = self.criterion(logits, labels)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = total_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc, all_predictions, all_labels
    
    def train(self, train_loader, val_loader):
        """Training loop chính"""
        best_val_acc = 0
        patience_counter = 0
        
        print(f"Starting training for {self.config.get('epochs', 100)} epochs...")
        
        for epoch in range(self.config.get('epochs', 100)):
            print(f"\nEpoch {epoch+1}/{self.config.get('epochs', 100)}")
            print("-" * 50)
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc, val_preds, val_labels = self.validate_epoch(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self.save_checkpoint(f"best_model_epoch_{epoch+1}.pth", epoch, val_loss)
                print(f"New best model saved! Val Acc: {val_acc:.2f}%")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config.get('early_stopping_patience', 10):
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # Log to wandb if available
            if 'wandb' in globals():
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
        
        print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")
        return best_val_acc
    
    def save_checkpoint(self, filename, epoch, loss):
        """Lưu checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.config
        }
        
        if hasattr(self.model, 'classification_head'):
            checkpoint['classification_head_state_dict'] = self.model.classification_head.state_dict()
        
        os.makedirs('models/species_recognition', exist_ok=True)
        torch.save(checkpoint, f'models/species_recognition/{filename}')
    
    def save_training_history(self):
        """Lưu training history"""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        with open('models/species_recognition/training_history.json', 'w') as f:
            json.dump(history, f, indent=2)


def main():
    """Main training function"""
    print("Plant Species Recognition Training")
    print("=" * 50)
    
    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Training config
    training_config = {
        'model_type': 'clip',  # 'cnn', 'vit', 'clip'
        'num_classes': 80000,  # PlantCLEF 2022
        'batch_size': 32,
        'learning_rate': 0.001,
        'weight_decay': 0.01,
        'epochs': 100,
        'early_stopping_patience': 10
    }
    
    # Initialize trainer
    trainer = SpeciesRecognitionTrainer(training_config)
    
    # Load datasets (sử dụng sample data nếu không có PlantCLEF)
    try:
        print("Loading PlantCLEF 2022 dataset...")
        train_loader, train_dataset = load_plantclef_dataset(
            "data/plantclef2022", "train", training_config['batch_size']
        )
        val_loader, val_dataset = load_plantclef_dataset(
            "data/plantclef2022", "val", training_config['batch_size']
        )
        print(f"Dataset loaded: {len(train_dataset)} train, {len(val_dataset)} val samples")
    except Exception as e:
        print(f"Error loading PlantCLEF dataset: {e}")
        print("Using sample data for demonstration...")
        # Tạo sample data
        from torch.utils.data import TensorDataset
        import torch
        
        # Tạo sample data
        n_samples = 1000
        n_classes = 10
        
        X = torch.randn(n_samples, 3, 224, 224)
        y = torch.randint(0, n_classes, (n_samples,))
        
        train_dataset = TensorDataset(X[:800], y[:800])
        val_dataset = TensorDataset(X[800:], y[800:])
        
        train_loader = DataLoader(train_dataset, batch_size=training_config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=training_config['batch_size'], shuffle=False)
    
    # Start training
    try:
        best_acc = trainer.train(train_loader, val_loader)
        trainer.save_training_history()
        print(f"Training completed successfully! Best accuracy: {best_acc:.2f}%")
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()







