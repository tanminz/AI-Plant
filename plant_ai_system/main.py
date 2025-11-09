"""
Plant AI System - Main Application
Tích hợp Module A (Plant Species Recognition) và Module B (Plant Health Monitor)
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Optional
from datetime import datetime

# Import modules
from module_a_species_recognition.plant_species_classifier import PlantSpeciesClassifier, create_plant_species_classifier
from module_b_health_monitor.plant_health_monitor import PlantHealthMonitor, create_plant_health_monitor


class PlantAISystem:
    """
    Hệ thống Plant AI chính tích hợp cả hai module
    """
    
    def __init__(self, config_path: str = "config.json"):
        """
        Khởi tạo Plant AI System
        
        Args:
            config_path: Đường dẫn đến file config
        """
        self.config = self._load_config(config_path)
        
        # Khởi tạo các module
        self.species_classifier = create_plant_species_classifier(
            model_type=self.config.get('species_model_type', 'clip')
        )
        
        self.health_monitor = create_plant_health_monitor(
            yolo_model_path=self.config.get('yolo_model_path')
        )
        
        print("Plant AI System initialized successfully!")
        print(f"Species Recognition Model: {self.config.get('species_model_type', 'clip')}")
        print(f"Health Monitor: YOLOv8 + Mask R-CNN")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration từ file"""
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Default config
            return {
                'species_model_type': 'clip',
                'yolo_model_path': None,
                'confidence_threshold': 0.5,
                'top_k_species': 5,
                'output_dir': 'results'
            }
    
    def analyze_plant(self, image_path: str, environmental_data: Dict = None) -> Dict:
        """
        Phân tích toàn diện cây trồng
        
        Args:
            image_path: Đường dẫn đến ảnh cây trồng
            environmental_data: Dữ liệu môi trường (nhiệt độ, độ ẩm, pH, etc.)
            
        Returns:
            Dict chứa kết quả phân tích toàn diện
        """
        print(f"Analyzing plant image: {image_path}")
        
        # 1. Nhận dạng loài thực vật (Module A)
        print("Step 1: Species Recognition...")
        species_results = self.species_classifier.predict(
            image_path, 
            top_k=self.config.get('top_k_species', 5)
        )
        
        # 2. Phân tích sức khỏe (Module B)
        print("Step 2: Health Analysis...")
        health_results = self.health_monitor.comprehensive_health_analysis(
            image_path,
            environmental_data
        )
        
        # 3. Tích hợp kết quả
        comprehensive_analysis = {
            'timestamp': datetime.now().isoformat(),
            'image_path': image_path,
            'species_recognition': {
                'top_species': species_results,
                'most_likely_species': species_results[0] if species_results else None
            },
            'health_analysis': health_results,
            'integrated_recommendations': self._generate_integrated_recommendations(
                species_results, health_results, environmental_data
            )
        }
        
        return comprehensive_analysis
    
    def _generate_integrated_recommendations(self, species_results: List[Dict], 
                                           health_results: Dict, 
                                           environmental_data: Dict = None) -> List[str]:
        """
        Tạo khuyến nghị tích hợp dựa trên loài và tình trạng sức khỏe
        
        Args:
            species_results: Kết quả nhận dạng loài
            health_results: Kết quả phân tích sức khỏe
            environmental_data: Dữ liệu môi trường
            
        Returns:
            List khuyến nghị tích hợp
        """
        recommendations = []
        
        # Khuyến nghị dựa trên loài
        if species_results:
            most_likely_species = species_results[0]
            species_name = most_likely_species.get('species_name', 'Unknown')
            confidence = most_likely_species.get('confidence', 0)
            
            if confidence > 0.8:
                recommendations.append(f"Cây được xác định là {species_name} với độ tin cậy cao ({confidence:.2f})")
            elif confidence > 0.5:
                recommendations.append(f"Cây có thể là {species_name} với độ tin cậy trung bình ({confidence:.2f})")
            else:
                recommendations.append(f"Không thể xác định chính xác loài cây (độ tin cậy thấp: {confidence:.2f})")
        
        # Khuyến nghị dựa trên sức khỏe
        health_score = health_results.get('health_score', 0)
        overall_status = health_results.get('overall_status', 'Unknown')
        
        if overall_status == "Critical":
            recommendations.append("⚠️ CẢNH BÁO: Cây trồng đang trong tình trạng nguy kịch, cần xử lý ngay lập tức!")
        elif overall_status == "Poor":
            recommendations.append("⚠️ Cây trồng đang trong tình trạng kém, cần chăm sóc đặc biệt")
        elif overall_status == "Fair":
            recommendations.append("ℹ️ Cây trồng ở tình trạng trung bình, cần theo dõi và cải thiện")
        elif overall_status == "Good":
            recommendations.append("✅ Cây trồng đang phát triển tốt, tiếp tục duy trì")
        elif overall_status == "Excellent":
            recommendations.append("🌟 Cây trồng đang phát triển xuất sắc!")
        
        # Thêm khuyến nghị từ health monitor
        health_recommendations = health_results.get('recommendations', [])
        recommendations.extend(health_recommendations)
        
        # Khuyến nghị dựa trên môi trường
        if environmental_data:
            temp = environmental_data.get('temperature', 25)
            humidity = environmental_data.get('humidity', 50)
            
            if temp < 15:
                recommendations.append("🌡️ Nhiệt độ quá thấp, cần tăng nhiệt độ môi trường")
            elif temp > 35:
                recommendations.append("🌡️ Nhiệt độ quá cao, cần giảm nhiệt độ")
            
            if humidity < 30:
                recommendations.append("💧 Độ ẩm quá thấp, cần tăng độ ẩm")
            elif humidity > 80:
                recommendations.append("💧 Độ ẩm quá cao, cần giảm độ ẩm")
        
        return recommendations
    
    def batch_analysis(self, image_dir: str, output_dir: str = None) -> Dict:
        """
        Phân tích hàng loạt ảnh
        
        Args:
            image_dir: Thư mục chứa ảnh
            output_dir: Thư mục lưu kết quả
            
        Returns:
            Dict chứa kết quả phân tích hàng loạt
        """
        if output_dir is None:
            output_dir = self.config.get('output_dir', 'results')
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Tìm tất cả ảnh trong thư mục
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for file in os.listdir(image_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(image_dir, file))
        
        print(f"Found {len(image_files)} images to analyze")
        
        batch_results = {
            'timestamp': datetime.now().isoformat(),
            'total_images': len(image_files),
            'results': []
        }
        
        for i, image_path in enumerate(image_files):
            print(f"Processing image {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
            
            try:
                analysis_result = self.analyze_plant(image_path)
                batch_results['results'].append(analysis_result)
                
                # Lưu kết quả riêng lẻ
                output_file = os.path.join(output_dir, f"analysis_{os.path.splitext(os.path.basename(image_path))[0]}.json")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(analysis_result, f, indent=2, ensure_ascii=False)
                
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                batch_results['results'].append({
                    'image_path': image_path,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        # Lưu kết quả tổng hợp
        batch_output_file = os.path.join(output_dir, 'batch_analysis_results.json')
        with open(batch_output_file, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, indent=2, ensure_ascii=False)
        
        print(f"Batch analysis completed. Results saved to {output_dir}")
        return batch_results
    
    def save_config(self, config_path: str = "config.json"):
        """Lưu configuration hiện tại"""
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        print(f"Configuration saved to {config_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Plant AI System")
    parser.add_argument("--image", type=str, help="Path to plant image")
    parser.add_argument("--batch", type=str, help="Directory containing images for batch analysis")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    parser.add_argument("--output", type=str, help="Output directory for results")
    parser.add_argument("--env-data", type=str, help="Path to environmental data JSON file")
    
    args = parser.parse_args()
    
    # Khởi tạo hệ thống
    plant_ai = PlantAISystem(config_path=args.config)
    
    # Load environmental data nếu có
    environmental_data = None
    if args.env_data and os.path.exists(args.env_data):
        with open(args.env_data, 'r', encoding='utf-8') as f:
            environmental_data = json.load(f)
    
    if args.image:
        # Phân tích ảnh đơn lẻ
        print(f"Analyzing single image: {args.image}")
        result = plant_ai.analyze_plant(args.image, environmental_data)
        
        # In kết quả
        print("\n" + "="*50)
        print("PLANT ANALYSIS RESULTS")
        print("="*50)
        
        # Species recognition results
        species_info = result['species_recognition']
        print(f"\n🌿 SPECIES RECOGNITION:")
        for i, species in enumerate(species_info['top_species']):
            print(f"  {i+1}. {species['species_name']} (confidence: {species['confidence']:.3f})")
        
        # Health analysis results
        health_info = result['health_analysis']
        print(f"\n🏥 HEALTH ANALYSIS:")
        print(f"  Health Score: {health_info['health_score']}/100")
        print(f"  Overall Status: {health_info['overall_status']}")
        print(f"  Diseases Detected: {health_info['disease_analysis']['total_diseases']}")
        print(f"  Pests Detected: {health_info['pest_analysis']['total_pests']}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(result['integrated_recommendations']):
            print(f"  {i+1}. {rec}")
        
        # Lưu kết quả
        if args.output:
            os.makedirs(args.output, exist_ok=True)
            output_file = os.path.join(args.output, f"analysis_{os.path.splitext(os.path.basename(args.image))[0]}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to: {output_file}")
    
    elif args.batch:
        # Phân tích hàng loạt
        print(f"Batch analyzing images in: {args.batch}")
        output_dir = args.output or "results"
        result = plant_ai.batch_analysis(args.batch, output_dir)
        print(f"Batch analysis completed. Processed {result['total_images']} images.")
    
    else:
        print("Please specify either --image for single image analysis or --batch for batch analysis")
        print("Use --help for more information")


if __name__ == "__main__":
    main()








