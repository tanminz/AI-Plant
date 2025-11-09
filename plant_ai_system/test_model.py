"""
Test script cho Plant AI model đã train
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from PIL import Image
import json
import random
import numpy as np


class PlantModelTester:
    """Tester cho Plant AI model"""
    
    def __init__(self, model_path, classes):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = classes
        self.num_classes = len(classes)
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Classes: {self.classes}")
    
    def _load_model(self, model_path):
        """Load trained model"""
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def predict_image(self, image_path):
        """Dự đoán một ảnh"""
        try:
            # Load và preprocess ảnh
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Dự đoán
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = self.classes[predicted.item()]
            confidence_score = confidence.item()
            
            return {
                'predicted_class': predicted_class,
                'confidence': confidence_score,
                'all_probabilities': probabilities[0].cpu().numpy().tolist()
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def test_random_samples(self, dataset_path, num_samples=10):
        """Test với các ảnh ngẫu nhiên từ dataset"""
        print(f"Testing with {num_samples} random samples...")
        
        # Lấy danh sách ảnh ngẫu nhiên
        all_images = []
        for class_name in self.classes:
            if class_name == 'splits':  # Skip splits folder
                continue
            class_path = os.path.join(dataset_path, class_name)
            if os.path.exists(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        all_images.append((os.path.join(class_path, img_name), class_name))
        
        # Chọn ngẫu nhiên
        test_images = random.sample(all_images, min(num_samples, len(all_images)))
        
        results = []
        correct = 0
        
        for img_path, true_class in test_images:
            result = self.predict_image(img_path)
            if 'error' not in result:
                predicted_class = result['predicted_class']
                confidence = result['confidence']
                
                is_correct = predicted_class == true_class
                if is_correct:
                    correct += 1
                
                results.append({
                    'image_path': img_path,
                    'true_class': true_class,
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'correct': is_correct
                })
                
                print(f"Image: {os.path.basename(img_path)}")
                print(f"  True: {true_class}")
                print(f"  Predicted: {predicted_class} ({confidence:.3f})")
                print(f"  Correct: {'Yes' if is_correct else 'No'}")
                print()
        
        accuracy = correct / len(results) if results else 0
        print(f"Test Accuracy: {accuracy:.2%} ({correct}/{len(results)})")
        
        return results, accuracy
    
    def get_top_predictions(self, image_path, top_k=3):
        """Lấy top-k predictions"""
        result = self.predict_image(image_path)
        if 'error' in result:
            return result
        
        # Sort probabilities
        probs = result['all_probabilities']
        sorted_indices = np.argsort(probs)[::-1]
        
        top_predictions = []
        for i in range(min(top_k, len(self.classes))):
            idx = sorted_indices[i]
            top_predictions.append({
                'class': self.classes[idx],
                'probability': probs[idx]
            })
        
        return {
            'top_predictions': top_predictions,
            'best_prediction': result['predicted_class'],
            'confidence': result['confidence']
        }


def main():
    """Main test function"""
    print("Plant AI Model Testing")
    print("=" * 40)
    
    # Model path
    model_path = "models/best_plant_model.pth"
    dataset_path = "data/health_monitoring/plant_leaf_diseases"
    
    if not os.path.exists(model_path):
        print(f"Model not found at: {model_path}")
        print("Please train the model first using simple_train.py")
        return
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at: {dataset_path}")
        return
    
    # Load model info
    checkpoint = torch.load(model_path, map_location='cpu')
    classes = checkpoint['classes']
    
    print(f"Model classes: {classes}")
    
    # Initialize tester
    tester = PlantModelTester(model_path, classes)
    
    # Test với ảnh ngẫu nhiên
    print("\n1. Testing with random samples...")
    results, accuracy = tester.test_random_samples(dataset_path, num_samples=5)
    
    # Test với ảnh cụ thể (nếu có)
    print("\n2. Testing specific images...")
    test_images = []
    
    # Tìm một vài ảnh để test
    for class_name in classes[:3]:  # Test 3 classes đầu
        if class_name == 'splits':
            continue
        class_path = os.path.join(dataset_path, class_name)
        if os.path.exists(class_path):
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                test_images.append(os.path.join(class_path, images[0]))
    
    for img_path in test_images:
        print(f"\nTesting: {os.path.basename(img_path)}")
        result = tester.predict_image(img_path)
        
        if 'error' not in result:
            print(f"  Predicted: {result['predicted_class']}")
            print(f"  Confidence: {result['confidence']:.3f}")
            
            # Top 3 predictions
            top_predictions = tester.get_top_predictions(img_path, top_k=3)
            print(f"  Top 3 predictions:")
            for i, pred in enumerate(top_predictions['top_predictions']):
                print(f"    {i+1}. {pred['class']}: {pred['probability']:.3f}")
        else:
            print(f"  Error: {result['error']}")
    
    print(f"\n[SUCCESS] Model testing completed!")
    print(f"Overall test accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()







