"""
Auto demo script cho Plant AI System
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


class PlantAIAutoDemo:
    """Auto Plant AI System Demo"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.classes = None
        self.transform = None
        
        print("Plant AI System - Auto Demo")
        print("=" * 40)
        print(f"Device: {self.device}")
    
    def load_model(self, model_path="models/best_plant_model.pth"):
        """Load trained model"""
        if not os.path.exists(model_path):
            print(f"Model not found at: {model_path}")
            return False
        
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
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def analyze_plant_health(self, image_path):
        """Phân tích sức khỏe cây trồng"""
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
            
            # Phân loại theo loại bệnh
            if 'healthy' in predicted_class.lower():
                health_status = "Healthy"
                health_score = 95
                recommendations = [
                    "Plant is healthy",
                    "Continue current care conditions",
                    "Monitor regularly for early disease signs"
                ]
            elif 'scab' in predicted_class.lower():
                health_status = "Disease: Apple Scab"
                health_score = 60
                recommendations = [
                    "Treat apple scab with copper-based fungicide",
                    "Remove and destroy infected leaves",
                    "Improve ventilation and reduce humidity"
                ]
            elif 'black_rot' in predicted_class.lower():
                health_status = "Disease: Black Rot"
                health_score = 45
                recommendations = [
                    "Treat black rot with mancozeb fungicide",
                    "Prune and destroy infected parts",
                    "Improve drainage"
                ]
            elif 'rust' in predicted_class.lower():
                health_status = "Disease: Cedar Apple Rust"
                health_score = 50
                recommendations = [
                    "Treat cedar apple rust with sulfur-based fungicide",
                    "Remove intermediate hosts",
                    "Apply preventive spray in spring"
                ]
            elif 'powdery_mildew' in predicted_class.lower():
                health_status = "Disease: Powdery Mildew"
                health_score = 55
                recommendations = [
                    "Treat powdery mildew with sulfur-based fungicide",
                    "Improve ventilation and reduce humidity",
                    "Prune to increase light exposure"
                ]
            else:
                health_status = "Unknown Disease"
                health_score = 70
                recommendations = [
                    "Need further examination for accurate diagnosis",
                    "Monitor for disease symptoms",
                    "Consult agricultural expert"
                ]
            
            return {
                'image_path': image_path,
                'predicted_class': predicted_class,
                'confidence': confidence_score,
                'health_status': health_status,
                'health_score': health_score,
                'recommendations': recommendations,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def run_auto_demo(self, dataset_path, num_samples=5):
        """Chạy demo tự động"""
        print(f"\nRunning Auto Demo with {num_samples} samples...")
        print("=" * 60)
        
        # Lấy ảnh ngẫu nhiên
        all_images = []
        for class_name in self.classes:
            if class_name == 'splits':
                continue
            class_path = os.path.join(dataset_path, class_name)
            if os.path.exists(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        all_images.append((os.path.join(class_path, img_name), class_name))
        
        if not all_images:
            print("No images found in dataset!")
            return
        
        # Chọn ngẫu nhiên
        test_images = random.sample(all_images, min(num_samples, len(all_images)))
        
        results = []
        correct_predictions = 0
        
        for i, (img_path, true_class) in enumerate(test_images, 1):
            print(f"\n{i}. Analyzing: {os.path.basename(img_path)}")
            print(f"   True class: {true_class}")
            
            # Phân tích sức khỏe
            analysis = self.analyze_plant_health(img_path)
            
            if 'error' not in analysis:
                predicted_class = analysis['predicted_class']
                confidence = analysis['confidence']
                health_status = analysis['health_status']
                health_score = analysis['health_score']
                
                is_correct = predicted_class == true_class
                if is_correct:
                    correct_predictions += 1
                
                print(f"   Predicted: {predicted_class}")
                print(f"   Confidence: {confidence:.3f}")
                print(f"   Health Status: {health_status}")
                print(f"   Health Score: {health_score}/100")
                print(f"   Correct: {'Yes' if is_correct else 'No'}")
                
                # Top recommendations
                print(f"   Top Recommendations:")
                for j, rec in enumerate(analysis['recommendations'][:2], 1):
                    print(f"     {j}. {rec}")
                
                results.append({
                    'image': os.path.basename(img_path),
                    'true_class': true_class,
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'health_score': health_score,
                    'correct': is_correct
                })
            else:
                print(f"   Error: {analysis['error']}")
                results.append({
                    'image': os.path.basename(img_path),
                    'error': analysis['error']
                })
        
        # Summary
        print(f"\n" + "=" * 60)
        print("DEMO SUMMARY")
        print("=" * 60)
        print(f"Total samples analyzed: {len(results)}")
        print(f"Correct predictions: {correct_predictions}")
        print(f"Accuracy: {correct_predictions/len(results)*100:.1f}%")
        
        # Health statistics
        health_scores = [r.get('health_score', 0) for r in results if 'health_score' in r]
        if health_scores:
            avg_health_score = sum(health_scores) / len(health_scores)
            print(f"Average health score: {avg_health_score:.1f}/100")
        
        # Disease distribution
        disease_counts = {}
        for result in results:
            if 'predicted_class' in result:
                pred_class = result['predicted_class']
                disease_counts[pred_class] = disease_counts.get(pred_class, 0) + 1
        
        print(f"\nDisease Distribution:")
        for disease, count in sorted(disease_counts.items()):
            print(f"  {disease}: {count} samples")
        
        return results


def main():
    """Main auto demo function"""
    print("Plant AI System - Auto Demo")
    print("=" * 50)
    
    # Initialize demo
    demo = PlantAIAutoDemo()
    
    # Load model
    if not demo.load_model():
        print("Failed to load model. Please train first using simple_train.py")
        return
    
    # Dataset path
    dataset_path = "data/health_monitoring/plant_leaf_diseases"
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at: {dataset_path}")
        return
    
    # Run auto demo
    results = demo.run_auto_demo(dataset_path, num_samples=5)
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/demo_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[SUCCESS] Auto demo completed!")
    print(f"Results saved to: results/demo_results.json")
    print(f"Plant AI System is ready for production use!")


if __name__ == "__main__":
    main()
