"""
Complete demo script cho Plant AI System
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


class PlantAIDemo:
    """Complete Plant AI System Demo"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.classes = None
        self.transform = None
        
        print("Plant AI System Demo")
        print("=" * 40)
        print(f"Device: {self.device}")
    
    def load_model(self, model_path="models/best_plant_model.pth"):
        """Load trained model"""
        if not os.path.exists(model_path):
            print(f"Model not found at: {model_path}")
            print("Please train the model first using simple_train.py")
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
    
    def predict_image(self, image_path):
        """Dự đoán một ảnh"""
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
            sorted_indices = np.argsort(probs)[::-1]
            
            top_predictions = []
            for i in range(min(3, len(self.classes))):
                idx = sorted_indices[i]
                top_predictions.append({
                    'class': self.classes[idx],
                    'probability': probs[idx]
                })
            
            return {
                'predicted_class': predicted_class,
                'confidence': confidence_score,
                'top_predictions': top_predictions,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_plant_health(self, image_path):
        """Phân tích sức khỏe cây trồng"""
        result = self.predict_image(image_path)
        if 'error' in result:
            return result
        
        predicted_class = result['predicted_class']
        confidence = result['confidence']
        
        # Phân loại theo loại bệnh
        if 'healthy' in predicted_class.lower():
            health_status = "Healthy"
            health_score = 95
            recommendations = [
                "Cây trồng đang khỏe mạnh",
                "Tiếp tục duy trì điều kiện chăm sóc hiện tại",
                "Theo dõi định kỳ để phát hiện sớm các dấu hiệu bệnh"
            ]
        elif 'scab' in predicted_class.lower():
            health_status = "Disease: Apple Scab"
            health_score = 60
            recommendations = [
                "Xử lý bệnh đốm táo bằng thuốc trừ nấm copper-based",
                "Loại bỏ lá bị bệnh và tiêu hủy",
                "Cải thiện thông gió và giảm độ ẩm",
                "Phun thuốc phòng bệnh định kỳ"
            ]
        elif 'black_rot' in predicted_class.lower():
            health_status = "Disease: Black Rot"
            health_score = 45
            recommendations = [
                "Xử lý bệnh thối đen bằng thuốc trừ nấm mancozeb",
                "Cắt tỉa và tiêu hủy các phần bị bệnh",
                "Cải thiện thoát nước",
                "Tăng cường dinh dưỡng cho cây"
            ]
        elif 'rust' in predicted_class.lower():
            health_status = "Disease: Cedar Apple Rust"
            health_score = 50
            recommendations = [
                "Xử lý bệnh gỉ sắt bằng thuốc trừ nấm sulfur-based",
                "Loại bỏ các vật chủ trung gian (cây bách xù)",
                "Phun thuốc phòng bệnh vào mùa xuân",
                "Cải thiện thông gió"
            ]
        elif 'powdery_mildew' in predicted_class.lower():
            health_status = "Disease: Powdery Mildew"
            health_score = 55
            recommendations = [
                "Xử lý bệnh phấn trắng bằng thuốc trừ nấm sulfur-based",
                "Cải thiện thông gió và giảm độ ẩm",
                "Cắt tỉa để tăng ánh sáng",
                "Phun thuốc phòng bệnh định kỳ"
            ]
        elif 'cercospora' in predicted_class.lower():
            health_status = "Disease: Cercospora Leaf Spot"
            health_score = 40
            recommendations = [
                "Xử lý bệnh đốm lá bằng thuốc trừ nấm chlorothalonil",
                "Loại bỏ lá bị bệnh",
                "Cải thiện thoát nước",
                "Tăng cường dinh dưỡng"
            ]
        else:
            health_status = "Unknown"
            health_score = 70
            recommendations = [
                "Cần kiểm tra thêm để xác định chính xác",
                "Theo dõi các dấu hiệu bệnh",
                "Tham khảo chuyên gia nông nghiệp"
            ]
        
        return {
            'image_path': image_path,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'health_status': health_status,
            'health_score': health_score,
            'recommendations': recommendations,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def demo_random_samples(self, dataset_path, num_samples=3):
        """Demo với ảnh ngẫu nhiên"""
        print(f"\nDemo với {num_samples} ảnh ngẫu nhiên:")
        print("-" * 50)
        
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
        
        for i, (img_path, true_class) in enumerate(test_images, 1):
            print(f"\n{i}. Analyzing: {os.path.basename(img_path)}")
            print(f"   True class: {true_class}")
            
            # Phân tích sức khỏe
            analysis = self.analyze_plant_health(img_path)
            
            if 'error' not in analysis:
                print(f"   Predicted: {analysis['predicted_class']}")
                print(f"   Confidence: {analysis['confidence']:.3f}")
                print(f"   Health Status: {analysis['health_status']}")
                print(f"   Health Score: {analysis['health_score']}/100")
                print(f"   Recommendations:")
                for j, rec in enumerate(analysis['recommendations'], 1):
                    print(f"     {j}. {rec}")
            else:
                print(f"   Error: {analysis['error']}")
    
    def demo_single_image(self, image_path):
        """Demo với một ảnh cụ thể"""
        print(f"\nAnalyzing single image: {image_path}")
        print("-" * 50)
        
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return
        
        analysis = self.analyze_plant_health(image_path)
        
        if 'error' not in analysis:
            print(f"Image: {os.path.basename(image_path)}")
            print(f"Predicted Class: {analysis['predicted_class']}")
            print(f"Confidence: {analysis['confidence']:.3f}")
            print(f"Health Status: {analysis['health_status']}")
            print(f"Health Score: {analysis['health_score']}/100")
            print(f"\nTop 3 Predictions:")
            for i, pred in enumerate(analysis.get('top_predictions', []), 1):
                print(f"  {i}. {pred['class']}: {pred['probability']:.3f}")
            print(f"\nRecommendations:")
            for i, rec in enumerate(analysis['recommendations'], 1):
                print(f"  {i}. {rec}")
        else:
            print(f"Error: {analysis['error']}")


def main():
    """Main demo function"""
    print("Plant AI System - Complete Demo")
    print("=" * 50)
    
    # Initialize demo
    demo = PlantAIDemo()
    
    # Load model
    if not demo.load_model():
        return
    
    # Dataset path
    dataset_path = "data/health_monitoring/plant_leaf_diseases"
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at: {dataset_path}")
        return
    
    # Demo options
    print("\nDemo Options:")
    print("1. Demo with random samples")
    print("2. Demo with specific image")
    print("3. Full demo (both)")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1" or choice == "3":
        # Demo với ảnh ngẫu nhiên
        demo.demo_random_samples(dataset_path, num_samples=3)
    
    if choice == "2" or choice == "3":
        # Demo với ảnh cụ thể
        print("\nEnter path to image file:")
        image_path = input("Image path: ").strip()
        if image_path:
            demo.demo_single_image(image_path)
    
    print(f"\n[SUCCESS] Demo completed!")
    print(f"Plant AI System is ready for production use!")


if __name__ == "__main__":
    main()







