"""
Plant Health Monitor
Sử dụng YOLOv8 + Mask R-CNN để phát hiện bệnh lá, sâu hại và nấm mốc
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from ultralytics import YOLO
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.model_zoo import model_zoo
from detectron2.data import MetadataCatalog
from typing import Dict, List, Tuple, Optional, Union
import json
import os
from datetime import datetime


class PlantHealthMonitor:
    """
    Hệ thống giám sát sức khỏe thực vật
    Kết hợp YOLOv8 và Mask R-CNN để phát hiện bệnh tật và sâu hại
    """
    
    def __init__(self, yolo_model_path: str = None, mask_rcnn_config: str = None):
        """
        Khởi tạo Plant Health Monitor
        
        Args:
            yolo_model_path: Đường dẫn đến YOLOv8 model
            mask_rcnn_config: Đường dẫn đến Mask R-CNN config
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Khởi tạo YOLOv8 model
        self.yolo_model = None
        if yolo_model_path and os.path.exists(yolo_model_path):
            self.yolo_model = YOLO(yolo_model_path)
        else:
            # Sử dụng pre-trained YOLOv8
            self.yolo_model = YOLO('yolov8n.pt')  # nano version for speed
        
        # Khởi tạo Mask R-CNN
        self.mask_rcnn_predictor = None
        self._setup_mask_rcnn()
        
        # Định nghĩa các loại bệnh và sâu hại
        self.disease_classes = {
            0: "healthy",
            1: "leaf_spot",
            2: "powdery_mildew", 
            3: "rust",
            4: "blight",
            5: "mosaic_virus",
            6: "anthracnose",
            7: "scab",
            8: "canker",
            9: "wilt"
        }
        
        self.pest_classes = {
            0: "aphid",
            1: "whitefly",
            2: "spider_mite",
            3: "thrips",
            4: "caterpillar",
            5: "beetle",
            6: "scale_insect",
            7: "mealybug"
        }
    
    def _setup_mask_rcnn(self):
        """Thiết lập Mask R-CNN model"""
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.disease_classes) + len(self.pest_classes)
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.DEVICE = str(self.device)
        
        self.mask_rcnn_predictor = DefaultPredictor(cfg)
    
    def detect_diseases(self, image_path: str, confidence_threshold: float = 0.5) -> Dict:
        """
        Phát hiện bệnh lá trên ảnh
        
        Args:
            image_path: Đường dẫn đến ảnh
            confidence_threshold: Ngưỡng confidence
            
        Returns:
            Dict chứa thông tin bệnh được phát hiện
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image from {image_path}")
        
        # Sử dụng YOLOv8 để detect objects
        yolo_results = self.yolo_model(image, conf=confidence_threshold)
        
        # Sử dụng Mask R-CNN để segmentation
        mask_rcnn_outputs = self.mask_rcnn_predictor(image)
        
        diseases_detected = []
        
        # Xử lý kết quả YOLO
        for result in yolo_results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    if cls in self.disease_classes and conf >= confidence_threshold:
                        diseases_detected.append({
                            'disease_type': self.disease_classes[cls],
                            'confidence': conf,
                            'bbox': box.xyxy[0].cpu().numpy().tolist(),
                            'detection_method': 'yolo'
                        })
        
        # Xử lý kết quả Mask R-CNN
        instances = mask_rcnn_outputs["instances"]
        if len(instances) > 0:
            for i in range(len(instances)):
                pred_class = instances.pred_classes[i].item()
                pred_score = instances.scores[i].item()
                pred_mask = instances.pred_masks[i].cpu().numpy()
                
                if pred_class in self.disease_classes and pred_score >= confidence_threshold:
                    diseases_detected.append({
                        'disease_type': self.disease_classes[pred_class],
                        'confidence': pred_score,
                        'mask': pred_mask,
                        'detection_method': 'mask_rcnn'
                    })
        
        return {
            'image_path': image_path,
            'timestamp': datetime.now().isoformat(),
            'diseases_detected': diseases_detected,
            'total_diseases': len(diseases_detected)
        }
    
    def detect_pests(self, image_path: str, confidence_threshold: float = 0.5) -> Dict:
        """
        Phát hiện sâu hại trên ảnh
        
        Args:
            image_path: Đường dẫn đến ảnh
            confidence_threshold: Ngưỡng confidence
            
        Returns:
            Dict chứa thông tin sâu hại được phát hiện
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image from {image_path}")
        
        # Sử dụng YOLOv8 để detect pests
        yolo_results = self.yolo_model(image, conf=confidence_threshold)
        
        pests_detected = []
        
        # Xử lý kết quả YOLO
        for result in yolo_results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    if cls in self.pest_classes and conf >= confidence_threshold:
                        pests_detected.append({
                            'pest_type': self.pest_classes[cls],
                            'confidence': conf,
                            'bbox': box.xyxy[0].cpu().numpy().tolist(),
                            'detection_method': 'yolo'
                        })
        
        return {
            'image_path': image_path,
            'timestamp': datetime.now().isoformat(),
            'pests_detected': pests_detected,
            'total_pests': len(pests_detected)
        }
    
    def comprehensive_health_analysis(self, image_path: str, 
                                    environmental_data: Dict = None) -> Dict:
        """
        Phân tích toàn diện sức khỏe thực vật
        
        Args:
            image_path: Đường dẫn đến ảnh
            environmental_data: Metadata môi trường (nhiệt độ, độ ẩm, pH, etc.)
            
        Returns:
            Dict chứa kết quả phân tích toàn diện
        """
        # Phát hiện bệnh
        disease_results = self.detect_diseases(image_path)
        
        # Phát hiện sâu hại
        pest_results = self.detect_pests(image_path)
        
        # Tính toán health score
        health_score = self._calculate_health_score(disease_results, pest_results, environmental_data)
        
        # Đưa ra khuyến nghị
        recommendations = self._generate_recommendations(disease_results, pest_results, environmental_data)
        
        return {
            'image_path': image_path,
            'timestamp': datetime.now().isoformat(),
            'disease_analysis': disease_results,
            'pest_analysis': pest_results,
            'health_score': health_score,
            'environmental_data': environmental_data,
            'recommendations': recommendations,
            'overall_status': self._determine_overall_status(health_score)
        }
    
    def _calculate_health_score(self, disease_results: Dict, pest_results: Dict, 
                             environmental_data: Dict = None) -> float:
        """
        Tính toán health score từ 0-100
        
        Args:
            disease_results: Kết quả phát hiện bệnh
            pest_results: Kết quả phát hiện sâu hại
            environmental_data: Dữ liệu môi trường
            
        Returns:
            Health score (0-100)
        """
        base_score = 100.0
        
        # Trừ điểm cho bệnh
        disease_penalty = len(disease_results['diseases_detected']) * 10
        for disease in disease_results['diseases_detected']:
            disease_penalty += disease['confidence'] * 20
        
        # Trừ điểm cho sâu hại
        pest_penalty = len(pest_results['pests_detected']) * 15
        for pest in pest_results['pests_detected']:
            pest_penalty += pest['confidence'] * 25
        
        # Điều chỉnh theo môi trường
        env_penalty = 0
        if environmental_data:
            # Nhiệt độ không phù hợp
            temp = environmental_data.get('temperature', 25)
            if temp < 15 or temp > 35:
                env_penalty += 10
            
            # Độ ẩm không phù hợp
            humidity = environmental_data.get('humidity', 50)
            if humidity < 30 or humidity > 80:
                env_penalty += 10
            
            # pH không phù hợp
            ph = environmental_data.get('ph', 6.5)
            if ph < 5.5 or ph > 7.5:
                env_penalty += 15
        
        health_score = max(0, base_score - disease_penalty - pest_penalty - env_penalty)
        return round(health_score, 2)
    
    def _generate_recommendations(self, disease_results: Dict, pest_results: Dict,
                                environmental_data: Dict = None) -> List[str]:
        """
        Tạo khuyến nghị dựa trên kết quả phân tích
        
        Args:
            disease_results: Kết quả phát hiện bệnh
            pest_results: Kết quả phát hiện sâu hại
            environmental_data: Dữ liệu môi trường
            
        Returns:
            List các khuyến nghị
        """
        recommendations = []
        
        # Khuyến nghị cho bệnh
        for disease in disease_results['diseases_detected']:
            disease_type = disease['disease_type']
            if disease_type == "leaf_spot":
                recommendations.append("Xử lý bệnh đốm lá bằng thuốc trừ nấm copper-based")
            elif disease_type == "powdery_mildew":
                recommendations.append("Xử lý bệnh phấn trắng bằng thuốc trừ nấm sulfur-based")
            elif disease_type == "rust":
                recommendations.append("Xử lý bệnh gỉ sắt bằng thuốc trừ nấm mancozeb")
            elif disease_type == "blight":
                recommendations.append("Xử lý bệnh héo xanh bằng thuốc trừ nấm chlorothalonil")
        
        # Khuyến nghị cho sâu hại
        for pest in pest_results['pests_detected']:
            pest_type = pest['pest_type']
            if pest_type == "aphid":
                recommendations.append("Xử lý rệp bằng thuốc trừ sâu neem oil hoặc pyrethrin")
            elif pest_type == "whitefly":
                recommendations.append("Xử lý ruồi trắng bằng thuốc trừ sâu imidacloprid")
            elif pest_type == "spider_mite":
                recommendations.append("Xử lý nhện đỏ bằng thuốc trừ sâu abamectin")
        
        # Khuyến nghị môi trường
        if environmental_data:
            temp = environmental_data.get('temperature', 25)
            if temp < 15:
                recommendations.append("Nhiệt độ quá thấp, cần tăng nhiệt độ môi trường")
            elif temp > 35:
                recommendations.append("Nhiệt độ quá cao, cần giảm nhiệt độ và tăng độ ẩm")
            
            humidity = environmental_data.get('humidity', 50)
            if humidity < 30:
                recommendations.append("Độ ẩm quá thấp, cần tăng độ ẩm môi trường")
            elif humidity > 80:
                recommendations.append("Độ ẩm quá cao, cần giảm độ ẩm để tránh nấm mốc")
        
        if not recommendations:
            recommendations.append("Cây trồng khỏe mạnh, tiếp tục duy trì điều kiện hiện tại")
        
        return recommendations
    
    def _determine_overall_status(self, health_score: float) -> str:
        """
        Xác định trạng thái tổng thể dựa trên health score
        
        Args:
            health_score: Điểm sức khỏe (0-100)
            
        Returns:
            Trạng thái tổng thể
        """
        if health_score >= 90:
            return "Excellent"
        elif health_score >= 75:
            return "Good"
        elif health_score >= 60:
            return "Fair"
        elif health_score >= 40:
            return "Poor"
        else:
            return "Critical"
    
    def save_analysis_report(self, analysis_results: Dict, output_path: str):
        """
        Lưu báo cáo phân tích
        
        Args:
            analysis_results: Kết quả phân tích
            output_path: Đường dẫn lưu file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        print(f"Analysis report saved to {output_path}")


def create_plant_health_monitor(yolo_model_path: str = None) -> PlantHealthMonitor:
    """
    Factory function để tạo PlantHealthMonitor
    
    Args:
        yolo_model_path: Đường dẫn đến YOLOv8 model (optional)
        
    Returns:
        PlantHealthMonitor instance
    """
    return PlantHealthMonitor(yolo_model_path=yolo_model_path)


if __name__ == "__main__":
    # Test health monitor
    monitor = create_plant_health_monitor()
    print("Plant Health Monitor initialized")
    print(f"Device: {monitor.device}")
    print(f"Disease classes: {list(monitor.disease_classes.values())}")
    print(f"Pest classes: {list(monitor.pest_classes.values())}")








