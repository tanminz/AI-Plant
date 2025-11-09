"""
Demo script cho Plant AI System
"""

import os
import json
import sys
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from datetime import datetime

# Add parent directory to path để import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plant_ai_system.main import PlantAISystem


def create_sample_plant_image(output_path: str = "sample_plant.jpg"):
    """
    Tạo ảnh mẫu cho demo
    
    Args:
        output_path: Đường dẫn lưu ảnh mẫu
    """
    # Tạo ảnh mẫu với màu xanh lá cây
    img = Image.new('RGB', (400, 300), color='lightgreen')
    draw = ImageDraw.Draw(img)
    
    # Vẽ một cây đơn giản
    # Thân cây
    draw.rectangle([190, 200, 210, 280], fill='brown')
    
    # Lá cây
    draw.ellipse([150, 150, 250, 220], fill='darkgreen')
    draw.ellipse([160, 130, 240, 200], fill='green')
    draw.ellipse([170, 110, 230, 180], fill='lightgreen')
    
    # Thêm một số đốm lá (bệnh)
    draw.ellipse([180, 160, 190, 170], fill='brown')
    draw.ellipse([200, 140, 210, 150], fill='brown')
    
    # Lưu ảnh
    img.save(output_path)
    print(f"Sample plant image created: {output_path}")
    return output_path


def create_sample_environmental_data(output_path: str = "sample_env_data.json"):
    """
    Tạo dữ liệu môi trường mẫu
    
    Args:
        output_path: Đường dẫn lưu file JSON
    """
    env_data = {
        "temperature": 25.5,
        "humidity": 65.0,
        "ph": 6.8,
        "light_intensity": 1200,
        "soil_moisture": 70.0,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(env_data, f, indent=2, ensure_ascii=False)
    
    print(f"Sample environmental data created: {output_path}")
    return output_path


def run_demo():
    """Chạy demo Plant AI System"""
    print("🌱 Plant AI System Demo")
    print("=" * 50)
    
    # Tạo dữ liệu mẫu
    print("\n1. Creating sample data...")
    sample_image = create_sample_plant_image()
    sample_env_data = create_sample_environmental_data()
    
    # Khởi tạo hệ thống
    print("\n2. Initializing Plant AI System...")
    try:
        plant_ai = PlantAISystem()
        print("✅ Plant AI System initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing Plant AI System: {e}")
        return
    
    # Load environmental data
    print("\n3. Loading environmental data...")
    try:
        with open(sample_env_data, 'r', encoding='utf-8') as f:
            environmental_data = json.load(f)
        print("✅ Environmental data loaded successfully!")
        print(f"   Temperature: {environmental_data['temperature']}°C")
        print(f"   Humidity: {environmental_data['humidity']}%")
        print(f"   pH: {environmental_data['ph']}")
    except Exception as e:
        print(f"❌ Error loading environmental data: {e}")
        environmental_data = None
    
    # Phân tích cây trồng
    print("\n4. Analyzing plant...")
    try:
        result = plant_ai.analyze_plant(sample_image, environmental_data)
        print("✅ Plant analysis completed successfully!")
        
        # Hiển thị kết quả
        print("\n" + "=" * 50)
        print("📊 ANALYSIS RESULTS")
        print("=" * 50)
        
        # Species recognition results
        species_info = result['species_recognition']
        print(f"\n🌿 SPECIES RECOGNITION:")
        if species_info['top_species']:
            for i, species in enumerate(species_info['top_species'][:3]):
                print(f"   {i+1}. {species['species_name']} (confidence: {species['confidence']:.3f})")
        else:
            print("   No species detected")
        
        # Health analysis results
        health_info = result['health_analysis']
        print(f"\n🏥 HEALTH ANALYSIS:")
        print(f"   Health Score: {health_info['health_score']}/100")
        print(f"   Overall Status: {health_info['overall_status']}")
        print(f"   Diseases Detected: {health_info['disease_analysis']['total_diseases']}")
        print(f"   Pests Detected: {health_info['pest_analysis']['total_pests']}")
        
        # Detailed disease analysis
        if health_info['disease_analysis']['diseases_detected']:
            print(f"\n🔍 DISEASE DETAILS:")
            for disease in health_info['disease_analysis']['diseases_detected']:
                print(f"   - {disease['disease_type']} (confidence: {disease['confidence']:.3f})")
        
        # Detailed pest analysis
        if health_info['pest_analysis']['pests_detected']:
            print(f"\n🐛 PEST DETAILS:")
            for pest in health_info['pest_analysis']['pests_detected']:
                print(f"   - {pest['pest_type']} (confidence: {pest['confidence']:.3f})")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(result['integrated_recommendations']):
            print(f"   {i+1}. {rec}")
        
        # Lưu kết quả
        output_file = "demo_analysis_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error during plant analysis: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup
    print(f"\n5. Cleaning up sample files...")
    try:
        os.remove(sample_image)
        os.remove(sample_env_data)
        print("✅ Sample files cleaned up")
    except:
        pass
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"Plant AI System is ready for use!")


def run_batch_demo():
    """Demo phân tích hàng loạt"""
    print("🌱 Plant AI System - Batch Analysis Demo")
    print("=" * 50)
    
    # Tạo thư mục ảnh mẫu
    batch_dir = "demo_images"
    os.makedirs(batch_dir, exist_ok=True)
    
    print(f"\n1. Creating sample images in {batch_dir}/...")
    for i in range(3):
        img_path = os.path.join(batch_dir, f"plant_{i+1}.jpg")
        create_sample_plant_image(img_path)
    
    # Khởi tạo hệ thống
    print(f"\n2. Initializing Plant AI System...")
    try:
        plant_ai = PlantAISystem()
        print("✅ Plant AI System initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing Plant AI System: {e}")
        return
    
    # Phân tích hàng loạt
    print(f"\n3. Running batch analysis...")
    try:
        result = plant_ai.batch_analysis(batch_dir, "demo_results")
        print("✅ Batch analysis completed successfully!")
        print(f"   Total images processed: {result['total_images']}")
        print(f"   Results saved to: demo_results/")
    except Exception as e:
        print(f"❌ Error during batch analysis: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup
    print(f"\n4. Cleaning up...")
    try:
        import shutil
        shutil.rmtree(batch_dir)
        print("✅ Demo images cleaned up")
    except:
        pass
    
    print(f"\n🎉 Batch demo completed successfully!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Plant AI System Demo")
    parser.add_argument("--mode", choices=["single", "batch"], default="single",
                       help="Demo mode: single image or batch analysis")
    
    args = parser.parse_args()
    
    if args.mode == "single":
        run_demo()
    elif args.mode == "batch":
        run_batch_demo()
    else:
        print("Invalid mode. Use --mode single or --mode batch")








