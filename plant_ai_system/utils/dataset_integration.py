"""
Script để tích hợp dataset Plant Leaf Diseases vào Plant AI System
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List
import shutil


def analyze_plant_leaf_diseases_dataset(data_path: str) -> Dict:
    """
    Phân tích dataset Plant Leaf Diseases
    
    Args:
        data_path: Đường dẫn đến dataset
        
    Returns:
        Dict chứa thông tin phân tích dataset
    """
    dataset_info = {
        'total_classes': 0,
        'total_images': 0,
        'classes': {},
        'class_distribution': {},
        'diseases': [],
        'healthy_plants': [],
        'background': []
    }
    
    # Duyệt qua các thư mục con
    for class_dir in os.listdir(data_path):
        class_path = os.path.join(data_path, class_dir)
        if os.path.isdir(class_path):
            # Đếm số ảnh trong class
            image_count = len([f for f in os.listdir(class_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            
            dataset_info['total_images'] += image_count
            dataset_info['classes'][class_dir] = image_count
            dataset_info['class_distribution'][class_dir] = image_count
            
            # Phân loại theo loại bệnh
            if 'healthy' in class_dir.lower():
                dataset_info['healthy_plants'].append({
                    'class': class_dir,
                    'count': image_count
                })
            elif 'background' in class_dir.lower():
                dataset_info['background'].append({
                    'class': class_dir,
                    'count': image_count
                })
            else:
                dataset_info['diseases'].append({
                    'class': class_dir,
                    'count': image_count
                })
    
    dataset_info['total_classes'] = len(dataset_info['classes'])
    
    return dataset_info


def create_dataset_metadata(data_path: str, output_path: str):
    """
    Tạo metadata cho dataset
    
    Args:
        data_path: Đường dẫn đến dataset
        output_path: Đường dẫn lưu metadata
    """
    print("Analyzing Plant Leaf Diseases dataset...")
    dataset_info = analyze_plant_leaf_diseases_dataset(data_path)
    
    # Tạo mapping cho disease classes
    disease_mapping = {}
    class_id = 0
    
    for class_name in dataset_info['classes'].keys():
        disease_mapping[class_id] = {
            'name': class_name,
            'type': 'disease' if 'healthy' not in class_name.lower() and 'background' not in class_name.lower() else 
                   'healthy' if 'healthy' in class_name.lower() else 'background',
            'count': dataset_info['classes'][class_name]
        }
        class_id += 1
    
    # Tạo metadata hoàn chỉnh
    metadata = {
        'dataset_name': 'Plant Leaf Diseases Dataset',
        'description': 'Dataset for identification of plant leaf diseases using deep learning',
        'total_classes': dataset_info['total_classes'],
        'total_images': dataset_info['total_images'],
        'classes': dataset_info['classes'],
        'class_mapping': disease_mapping,
        'statistics': {
            'diseases': len(dataset_info['diseases']),
            'healthy_plants': len(dataset_info['healthy_plants']),
            'background': len(dataset_info['background'])
        },
        'class_distribution': dataset_info['class_distribution']
    }
    
    # Lưu metadata
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"Dataset metadata saved to: {output_path}")
    return metadata


def create_training_splits(data_path: str, output_dir: str, train_ratio: float = 0.7, 
                         val_ratio: float = 0.15, test_ratio: float = 0.15):
    """
    Tạo train/val/test splits cho dataset
    
    Args:
        data_path: Đường dẫn đến dataset
        output_dir: Thư mục lưu splits
        train_ratio: Tỷ lệ training set
        val_ratio: Tỷ lệ validation set  
        test_ratio: Tỷ lệ test set
    """
    import random
    from sklearn.model_selection import train_test_split
    
    # Tạo thư mục splits
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)
    
    splits_info = {
        'train': [],
        'val': [],
        'test': []
    }
    
    # Duyệt qua từng class
    for class_name in os.listdir(data_path):
        class_path = os.path.join(data_path, class_name)
        if os.path.isdir(class_path):
            # Lấy danh sách ảnh
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Shuffle images
            random.shuffle(images)
            
            # Split images
            n_images = len(images)
            n_train = int(n_images * train_ratio)
            n_val = int(n_images * val_ratio)
            
            train_images = images[:n_train]
            val_images = images[n_train:n_train + n_val]
            test_images = images[n_train + n_val:]
            
            # Copy images to respective directories
            for split, split_images in [('train', train_images), 
                                       ('val', val_images), 
                                       ('test', test_images)]:
                split_class_dir = os.path.join(output_dir, split, class_name)
                os.makedirs(split_class_dir, exist_ok=True)
                
                for img in split_images:
                    src = os.path.join(class_path, img)
                    dst = os.path.join(split_class_dir, img)
                    shutil.copy2(src, dst)
                
                splits_info[split].append({
                    'class': class_name,
                    'count': len(split_images)
                })
    
    # Lưu splits info
    with open(os.path.join(output_dir, 'splits_info.json'), 'w', encoding='utf-8') as f:
        json.dump(splits_info, f, indent=2, ensure_ascii=False)
    
    print(f"Dataset splits created in: {output_dir}")
    return splits_info


def update_plant_ai_config(dataset_path: str, config_path: str):
    """
    Cập nhật config của Plant AI System với dataset mới
    
    Args:
        dataset_path: Đường dẫn đến dataset
        config_path: Đường dẫn đến config file
    """
    # Load config hiện tại
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Cập nhật data paths
    config['data_paths']['plant_leaf_diseases'] = dataset_path
    config['data_paths']['health_monitoring'] = os.path.join(dataset_path, '..')
    
    # Cập nhật số classes cho health monitoring
    dataset_info = analyze_plant_leaf_diseases_dataset(dataset_path)
    config['health_monitoring'] = {
        'num_classes': dataset_info['total_classes'],
        'disease_classes': [cls for cls in dataset_info['classes'].keys() 
                           if 'healthy' not in cls.lower() and 'background' not in cls.lower()],
        'healthy_classes': [cls for cls in dataset_info['classes'].keys() 
                           if 'healthy' in cls.lower()],
        'background_classes': [cls for cls in dataset_info['classes'].keys() 
                              if 'background' in cls.lower()]
    }
    
    # Lưu config đã cập nhật
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"Plant AI config updated: {config_path}")


def main():
    """Main function"""
    # Đường dẫn dataset
    dataset_path = "data/health_monitoring/plant_leaf_diseases"
    config_path = "config.json"
    
    print("Plant AI System - Dataset Integration")
    print("=" * 50)
    
    # 1. Phân tích dataset
    print("\n1. Analyzing dataset...")
    dataset_info = analyze_plant_leaf_diseases_dataset(dataset_path)
    
    print(f"   Total classes: {dataset_info['total_classes']}")
    print(f"   Total images: {dataset_info['total_images']}")
    print(f"   Disease classes: {len(dataset_info['diseases'])}")
    print(f"   Healthy classes: {len(dataset_info['healthy_plants'])}")
    print(f"   Background classes: {len(dataset_info['background'])}")
    
    # 2. Tạo metadata
    print("\n2. Creating dataset metadata...")
    metadata_path = os.path.join(dataset_path, "dataset_metadata.json")
    metadata = create_dataset_metadata(dataset_path, metadata_path)
    
    # 3. Tạo training splits
    print("\n3. Creating training splits...")
    splits_dir = os.path.join(dataset_path, "splits")
    splits_info = create_training_splits(dataset_path, splits_dir)
    
    # 4. Cập nhật config
    print("\n4. Updating Plant AI config...")
    update_plant_ai_config(dataset_path, config_path)
    
    print("\n[OK] Dataset integration completed successfully!")
    print(f"   Dataset path: {dataset_path}")
    print(f"   Metadata: {metadata_path}")
    print(f"   Splits: {splits_dir}")
    print(f"   Config: {config_path}")


if __name__ == "__main__":
    main()
