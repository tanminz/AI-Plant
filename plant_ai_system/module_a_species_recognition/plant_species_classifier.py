"""
Plant Species Classifier
Hỗ trợ nhiều kiến trúc: CNN, ViT, CLIP-finetune
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import CLIPModel, CLIPProcessor
from torchvision.models import resnet50, vit_b_16
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import os


class PlantSpeciesClassifier:
    """
    Classifier chính cho nhận dạng loài thực vật
    Hỗ trợ CNN, ViT, và CLIP-finetune
    """
    
    def __init__(self, model_type: str = "clip", num_classes: int = 80000):
        """
        Khởi tạo classifier
        
        Args:
            model_type: Loại mô hình ("cnn", "vit", "clip")
            num_classes: Số lượng loài thực vật (PlantCLEF 2022: ~80,000)
        """
        self.model_type = model_type
        self.num_classes = num_classes
        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Khởi tạo mô hình dựa trên loại được chọn"""
        if self.model_type == "cnn":
            self._init_cnn_model()
        elif self.model_type == "vit":
            self._init_vit_model()
        elif self.model_type == "clip":
            self._init_clip_model()
        else:
            raise ValueError(f"Model type {self.model_type} not supported")
    
    def _init_cnn_model(self):
        """Khởi tạo mô hình CNN (ResNet50)"""
        self.model = resnet50(pretrained=True)
        # Thay đổi layer cuối cho số lượng classes
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        self.model = self.model.to(self.device)
        
        # Transform cho CNN
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _init_vit_model(self):
        """Khởi tạo mô hình Vision Transformer"""
        self.model = vit_b_16(pretrained=True)
        # Thay đổi head cho số lượng classes
        self.model.heads.head = nn.Linear(self.model.heads.head.in_features, self.num_classes)
        self.model = self.model.to(self.device)
        
        # Transform cho ViT
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _init_clip_model(self):
        """Khởi tạo mô hình CLIP"""
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Thêm classification head
        self.classification_head = nn.Linear(
            self.model.config.projection_dim, 
            self.num_classes
        ).to(self.device)
        
        self.model = self.model.to(self.device)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load mô hình từ checkpoint"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if hasattr(self, 'classification_head'):
                self.classification_head.load_state_dict(checkpoint['classification_head_state_dict'])
            print(f"Model loaded from {checkpoint_path}")
        else:
            print(f"Checkpoint not found at {checkpoint_path}")
    
    def save_checkpoint(self, checkpoint_path: str, epoch: int, loss: float):
        """Lưu checkpoint của mô hình"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'loss': loss,
        }
        
        if hasattr(self, 'classification_head'):
            checkpoint['classification_head_state_dict'] = self.classification_head.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def predict(self, image_path: str, top_k: int = 5) -> List[Dict]:
        """
        Dự đoán loài thực vật từ ảnh
        
        Args:
            image_path: Đường dẫn đến ảnh
            top_k: Số lượng kết quả top-k
            
        Returns:
            List các dict chứa species_id, confidence, species_name
        """
        self.model.eval()
        
        with torch.no_grad():
            if self.model_type == "clip":
                return self._predict_clip(image_path, top_k)
            else:
                return self._predict_cnn_vit(image_path, top_k)
    
    def _predict_cnn_vit(self, image_path: str, top_k: int) -> List[Dict]:
        """Dự đoán cho CNN và ViT"""
        from PIL import Image
        
        # Load và preprocess ảnh
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Dự đoán
        outputs = self.model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        
        # Lấy top-k predictions
        top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
        
        results = []
        for i in range(top_k):
            results.append({
                'species_id': top_indices[0][i].item(),
                'confidence': top_probs[0][i].item(),
                'species_name': f"Species_{top_indices[0][i].item()}"  # Cần mapping thực tế
            })
        
        return results
    
    def _predict_clip(self, image_path: str, top_k: int) -> List[Dict]:
        """Dự đoán cho CLIP"""
        from PIL import Image
        
        # Load ảnh
        image = Image.open(image_path).convert('RGB')
        
        # Preprocess với CLIP processor
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        # Lấy image features
        image_features = self.model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Classification
        logits = self.classification_head(image_features)
        probabilities = torch.softmax(logits, dim=1)
        
        # Lấy top-k predictions
        top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
        
        results = []
        for i in range(top_k):
            results.append({
                'species_id': top_indices[0][i].item(),
                'confidence': top_probs[0][i].item(),
                'species_name': f"Species_{top_indices[0][i].item()}"  # Cần mapping thực tế
            })
        
        return results
    
    def train_step(self, images: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Thực hiện một bước training
        
        Args:
            images: Batch ảnh
            labels: Batch labels
            
        Returns:
            Loss value
        """
        self.model.train()
        
        if self.model_type == "clip":
            # CLIP training với contrastive learning
            image_features = self.model.get_image_features(images)
            text_features = self.model.get_text_features(labels)  # Cần text prompts
            
            # Contrastive loss
            logits_per_image = image_features @ text_features.T
            logits_per_text = text_features @ image_features.T
            
            labels = torch.arange(len(images)).to(self.device)
            loss = (nn.CrossEntropyLoss()(logits_per_image, labels) + 
                   nn.CrossEntropyLoss()(logits_per_text, labels)) / 2
        else:
            # Standard classification
            outputs = self.model(images)
            loss = nn.CrossEntropyLoss()(outputs, labels)
        
        return loss.item()


class PlantCLEFDataLoader:
    """
    Data loader cho PlantCLEF 2022 dataset
    """
    
    def __init__(self, data_path: str, batch_size: int = 32):
        self.data_path = data_path
        self.batch_size = batch_size
        self.species_mapping = self._load_species_mapping()
    
    def _load_species_mapping(self) -> Dict:
        """Load mapping từ species_id đến species_name"""
        mapping_file = os.path.join(self.data_path, "species_mapping.json")
        if os.path.exists(mapping_file):
            with open(mapping_file, 'r') as f:
                return json.load(f)
        return {}
    
    def get_dataloader(self, split: str = "train"):
        """
        Tạo DataLoader cho split được chỉ định
        
        Args:
            split: "train", "val", hoặc "test"
        """
        # Implement data loading logic
        # Cần implement dựa trên cấu trúc PlantCLEF 2022
        pass


def create_plant_species_classifier(model_type: str = "clip") -> PlantSpeciesClassifier:
    """
    Factory function để tạo PlantSpeciesClassifier
    
    Args:
        model_type: Loại mô hình ("cnn", "vit", "clip")
        
    Returns:
        PlantSpeciesClassifier instance
    """
    return PlantSpeciesClassifier(model_type=model_type)


if __name__ == "__main__":
    # Test classifier
    classifier = create_plant_species_classifier("clip")
    print(f"Plant Species Classifier initialized with {classifier.model_type} model")
    print(f"Device: {classifier.device}")
    print(f"Number of classes: {classifier.num_classes}")








