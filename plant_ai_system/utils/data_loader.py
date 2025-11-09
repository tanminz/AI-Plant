"""
Data loading utilities for Plant AI System
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from typing import Dict, List, Tuple, Optional
import numpy as np


class PlantCLEFDataset(Dataset):
    """
    Dataset loader cho PlantCLEF 2022
    """
    
    def __init__(self, data_path: str, split: str = "train", 
                 transform: Optional[transforms.Compose] = None):
        """
        Khởi tạo PlantCLEF dataset
        
        Args:
            data_path: Đường dẫn đến dataset
            split: "train", "val", hoặc "test"
            transform: Transformations cho ảnh
        """
        self.data_path = data_path
        self.split = split
        self.transform = transform or self._get_default_transform()
        
        # Load metadata
        self.metadata = self._load_metadata()
        self.species_mapping = self._load_species_mapping()
        
    def _load_metadata(self) -> pd.DataFrame:
        """Load metadata từ CSV files"""
        metadata_file = os.path.join(self.data_path, f"{self.split}.csv")
        if os.path.exists(metadata_file):
            return pd.read_csv(metadata_file)
        else:
            # Tạo metadata mẫu nếu không có file
            return self._create_sample_metadata()
    
    def _load_species_mapping(self) -> Dict:
        """Load mapping từ species_id đến species_name"""
        mapping_file = os.path.join(self.data_path, "species_mapping.json")
        if os.path.exists(mapping_file):
            with open(mapping_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _create_sample_metadata(self) -> pd.DataFrame:
        """Tạo metadata mẫu cho testing"""
        # Tạo sample data
        sample_data = {
            'image_id': [f"sample_{i:06d}" for i in range(100)],
            'species_id': np.random.randint(0, 1000, 100),
            'image_path': [f"images/sample_{i:06d}.jpg" for i in range(100)]
        }
        return pd.DataFrame(sample_data)
    
    def _get_default_transform(self) -> transforms.Compose:
        """Transform mặc định cho ảnh"""
        if self.split == "train":
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Lấy item từ dataset
        
        Returns:
            Tuple (image_tensor, species_id)
        """
        row = self.metadata.iloc[idx]
        image_path = os.path.join(self.data_path, row['image_path'])
        
        # Load ảnh
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            # Tạo ảnh mẫu nếu không load được
            image = Image.new('RGB', (224, 224), color='green')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        species_id = int(row['species_id'])
        return image, species_id
    
    def get_species_name(self, species_id: int) -> str:
        """Lấy tên loài từ species_id"""
        return self.species_mapping.get(str(species_id), f"Species_{species_id}")


class HealthMonitoringDataset(Dataset):
    """
    Dataset cho health monitoring
    """
    
    def __init__(self, data_path: str, split: str = "train",
                 transform: Optional[transforms.Compose] = None):
        """
        Khởi tạo health monitoring dataset
        
        Args:
            data_path: Đường dẫn đến dataset
            split: "train", "val", hoặc "test"
            transform: Transformations cho ảnh
        """
        self.data_path = data_path
        self.split = split
        self.transform = transform or self._get_default_transform()
        
        # Load metadata
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> pd.DataFrame:
        """Load metadata cho health monitoring"""
        metadata_file = os.path.join(self.data_path, f"health_{self.split}.csv")
        if os.path.exists(metadata_file):
            return pd.read_csv(metadata_file)
        else:
            return self._create_sample_health_metadata()
    
    def _create_sample_health_metadata(self) -> pd.DataFrame:
        """Tạo metadata mẫu cho health monitoring"""
        sample_data = {
            'image_id': [f"health_{i:06d}" for i in range(50)],
            'image_path': [f"images/health_{i:06d}.jpg" for i in range(50)],
            'diseases': [np.random.choice(['healthy', 'leaf_spot', 'powdery_mildew'], 1)[0] for _ in range(50)],
            'pests': [np.random.choice(['none', 'aphid', 'whitefly'], 1)[0] for _ in range(50)],
            'health_score': np.random.uniform(20, 100, 50)
        }
        return pd.DataFrame(sample_data)
    
    def _get_default_transform(self) -> transforms.Compose:
        """Transform mặc định cho health monitoring"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Lấy item từ health monitoring dataset
        
        Returns:
            Dict chứa image, diseases, pests, health_score
        """
        row = self.metadata.iloc[idx]
        image_path = os.path.join(self.data_path, row['image_path'])
        
        # Load ảnh
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), color='green')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'diseases': row['diseases'],
            'pests': row['pests'],
            'health_score': float(row['health_score']),
            'image_id': row['image_id']
        }


def create_dataloader(dataset: Dataset, batch_size: int = 32, 
                     shuffle: bool = True, num_workers: int = 4) -> DataLoader:
    """
    Tạo DataLoader từ dataset
    
    Args:
        dataset: Dataset object
        batch_size: Kích thước batch
        shuffle: Có shuffle data không
        num_workers: Số worker processes
        
    Returns:
        DataLoader object
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


def load_plantclef_dataset(data_path: str, split: str = "train", 
                          batch_size: int = 32) -> Tuple[DataLoader, PlantCLEFDataset]:
    """
    Load PlantCLEF dataset
    
    Args:
        data_path: Đường dẫn đến dataset
        split: "train", "val", hoặc "test"
        batch_size: Kích thước batch
        
    Returns:
        Tuple (DataLoader, Dataset)
    """
    dataset = PlantCLEFDataset(data_path, split)
    dataloader = create_dataloader(dataset, batch_size, shuffle=(split=="train"))
    return dataloader, dataset


def load_health_monitoring_dataset(data_path: str, split: str = "train",
                                 batch_size: int = 32) -> Tuple[DataLoader, HealthMonitoringDataset]:
    """
    Load health monitoring dataset
    
    Args:
        data_path: Đường dẫn đến dataset
        split: "train", "val", hoặc "test"
        batch_size: Kích thước batch
        
    Returns:
        Tuple (DataLoader, Dataset)
    """
    dataset = HealthMonitoringDataset(data_path, split)
    dataloader = create_dataloader(dataset, batch_size, shuffle=(split=="train"))
    return dataloader, dataset


if __name__ == "__main__":
    # Test data loaders
    print("Testing PlantCLEF Dataset...")
    try:
        dataloader, dataset = load_plantclef_dataset("data/plantclef2022", "train", 16)
        print(f"PlantCLEF Dataset loaded: {len(dataset)} samples")
        
        # Test một batch
        for batch in dataloader:
            images, labels = batch
            print(f"Batch shape: {images.shape}, Labels shape: {labels.shape}")
            break
    except Exception as e:
        print(f"Error loading PlantCLEF dataset: {e}")
    
    print("\nTesting Health Monitoring Dataset...")
    try:
        dataloader, dataset = load_health_monitoring_dataset("data/health_monitoring", "train", 16)
        print(f"Health Monitoring Dataset loaded: {len(dataset)} samples")
        
        # Test một batch
        for batch in dataloader:
            print(f"Batch keys: {batch.keys()}")
            print(f"Image shape: {batch['image'].shape}")
            break
    except Exception as e:
        print(f"Error loading Health Monitoring dataset: {e}")








