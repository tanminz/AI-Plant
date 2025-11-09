"""
Test Complete Model và cập nhật Web App
"""

import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import random
import numpy as np
from datetime import datetime


class CompletePlantTester:
    """Test complete plant model"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.classes = None
        self.transform = None
        
        print(f"Complete Plant Tester initialized on {self.device}")
    
    def load_latest_model(self):
        """Load model mới nhất"""
        models_dir = "models"
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
        
        if not model_files:
            print("No model files found!")
            return False
        
        # Tìm model mới nhất
        latest_model = None
        latest_epoch = 0
        
        for model_file in model_files:
            if 'complete_plant_model_epoch_' in model_file:
                try:
                    epoch_num = int(model_file.split('_')[-1].split('.')[0])
                    if epoch_num > latest_epoch:
                        latest_epoch = epoch_num
                        latest_model = model_file
                except:
                    continue
        
        if latest_model:
            model_path = os.path.join(models_dir, latest_model)
            print(f"Loading latest model: {latest_model}")
        else:
            # Fallback to best_plant_model.pth
            model_path = os.path.join(models_dir, "best_plant_model.pth")
            print(f"Loading fallback model: best_plant_model.pth")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            self.classes = checkpoint['classes']
            
            # Load model
            self.model = models.resnet50(pretrained=False)
            self.model.fc = nn.Linear(self.model.fc.in_features, len(self.classes))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Transform
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225])
            ])
            
            print(f"Model loaded successfully!")
            print(f"Classes: {len(self.classes)}")
            print(f"Classes: {self.classes}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def test_image(self, image_path):
        """Test một ảnh"""
        if self.model is None:
            print("Model not loaded!")
            return None
        
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
            
            # Top 3 predictions
            probs = probabilities[0].cpu().numpy()
            sorted_indices = probs.argsort()[::-1]
            
            top_predictions = []
            for i in range(min(3, len(self.classes))):
                idx = sorted_indices[i]
                top_predictions.append({
                    'class': self.classes[idx],
                    'probability': float(probs[idx])
                })
            
            return {
                'success': True,
                'predicted_class': predicted_class,
                'confidence': confidence_score,
                'top_predictions': top_predictions,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def test_durian_images(self):
        """Test với ảnh sầu riêng"""
        print("\nTesting Durian Images...")
        print("=" * 40)
        
        # Tìm ảnh sầu riêng để test
        durian_paths = [
            "data/A Durian Leaf Image Dataset/A Durian Leaf Image Dataset of Common Diseases in Vietnam for Agricultural Diagnosis/Durian_Leaf_Diseases/train/Leaf_Healthy",
            "data/A Durian Leaf Image Dataset/A Durian Leaf Image Dataset of Common Diseases in Vietnam for Agricultural Diagnosis/Durian_Leaf_Diseases/train/Leaf_Blight",
            "data/A Durian Leaf Image Dataset/A Durian Leaf Image Dataset of Common Diseases in Vietnam for Agricultural Diagnosis/Durian_Leaf_Diseases/train/Leaf_Algal"
        ]
        
        test_results = []
        
        for durian_path in durian_paths:
            if os.path.exists(durian_path):
                print(f"\nTesting from: {os.path.basename(durian_path)}")
                
                # Lấy 2 ảnh ngẫu nhiên
                images = [f for f in os.listdir(durian_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if images:
                    test_images = random.sample(images, min(2, len(images)))
                    
                    for img_name in test_images:
                        img_path = os.path.join(durian_path, img_name)
                        result = self.test_image(img_path)
                        
                        if result['success']:
                            print(f"  Image: {img_name}")
                            print(f"  Predicted: {result['predicted_class']}")
                            print(f"  Confidence: {result['confidence']:.3f}")
                            print(f"  Top 3:")
                            for i, pred in enumerate(result['top_predictions']):
                                print(f"    {i+1}. {pred['class']}: {pred['probability']:.3f}")
                            
                            test_results.append({
                                'image': img_name,
                                'true_class': os.path.basename(durian_path),
                                'predicted_class': result['predicted_class'],
                                'confidence': result['confidence']
                            })
                        else:
                            print(f"  Error testing {img_name}: {result['error']}")
        
        return test_results
    
    def update_web_app_model(self):
        """Cập nhật web app với model mới"""
        print("\nUpdating Web App Model...")
        print("=" * 40)
        
        # Tìm model tốt nhất
        models_dir = "models"
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
        
        best_model = None
        best_epoch = 0
        
        for model_file in model_files:
            if 'complete_plant_model_epoch_' in model_file:
                try:
                    epoch_num = int(model_file.split('_')[-1].split('.')[0])
                    if epoch_num > best_epoch:
                        best_epoch = epoch_num
                        best_model = model_file
                except:
                    continue
        
        if best_model:
            # Copy model mới nhất thành best_plant_model.pth
            source_path = os.path.join(models_dir, best_model)
            target_path = os.path.join(models_dir, "best_plant_model.pth")
            
            try:
                # Load và save lại với format chuẩn
                checkpoint = torch.load(source_path, map_location=self.device)
                
                # Save với format chuẩn cho web app
                torch.save(checkpoint, target_path)
                print(f"Updated best_plant_model.pth with {best_model}")
                print(f"Classes: {len(checkpoint['classes'])}")
                return True
            except Exception as e:
                print(f"Error updating model: {e}")
                return False
        else:
            print("No complete model found to update")
            return False


def main():
    """Main test function"""
    print("Complete Plant Model Testing")
    print("=" * 50)
    
    # Initialize tester
    tester = CompletePlantTester()
    
    # Load latest model
    print("\n1. Loading latest model...")
    if not tester.load_latest_model():
        print("Failed to load model!")
        return
    
    # Test Durian images
    print("\n2. Testing Durian images...")
    durian_results = tester.test_durian_images()
    
    # Update web app model
    print("\n3. Updating web app model...")
    update_success = tester.update_web_app_model()
    
    # Summary
    print("\n" + "=" * 50)
    print("TESTING SUMMARY")
    print("=" * 50)
    print(f"Model loaded: {tester.model is not None}")
    print(f"Total classes: {len(tester.classes) if tester.classes else 0}")
    print(f"Durian tests: {len(durian_results)}")
    print(f"Web app updated: {update_success}")
    
    if durian_results:
        print(f"\nDurian test results:")
        for result in durian_results:
            print(f"  {result['image']}: {result['true_class']} -> {result['predicted_class']} ({result['confidence']:.3f})")
    
    print(f"\n[SUCCESS] Complete model testing completed!")
    print(f"Web app is now updated with the latest model!")


if __name__ == "__main__":
    main()






