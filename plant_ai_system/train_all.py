"""
Training script tổng hợp cho toàn bộ Plant AI System
"""

import os
import sys
import json
import time
from datetime import datetime
import argparse

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def train_species_recognition():
    """Training Module A - Species Recognition"""
    print("\n" + "="*60)
    print("MODULE A: PLANT SPECIES RECOGNITION TRAINING")
    print("="*60)
    
    try:
        from train_species_recognition import main as train_species_main
        train_species_main()
        return True
    except Exception as e:
        print(f"Species Recognition training failed: {e}")
        return False


def train_health_monitor():
    """Training Module B - Health Monitor"""
    print("\n" + "="*60)
    print("MODULE B: PLANT HEALTH MONITOR TRAINING")
    print("="*60)
    
    try:
        from train_health_monitor import main as train_health_main
        train_health_main()
        return True
    except Exception as e:
        print(f"Health Monitor training failed: {e}")
        return False


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Plant AI System Training")
    parser.add_argument("--module", choices=["species", "health", "all"], default="all",
                       help="Which module to train")
    parser.add_argument("--skip-species", action="store_true",
                       help="Skip species recognition training")
    parser.add_argument("--skip-health", action="store_true",
                       help="Skip health monitor training")
    
    args = parser.parse_args()
    
    print("🌱 Plant AI System - Complete Training Pipeline")
    print("=" * 60)
    print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target modules: {args.module}")
    
    # Load config
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        print(f"Config loaded successfully")
    except Exception as e:
        print(f"Error loading config: {e}")
        return
    
    # Training results
    training_results = {
        'timestamp': datetime.now().isoformat(),
        'species_recognition': {'status': 'not_started', 'success': False},
        'health_monitor': {'status': 'not_started', 'success': False},
        'total_time': 0
    }
    
    start_time = time.time()
    
    # Training Module A - Species Recognition
    if args.module in ["species", "all"] and not args.skip_species:
        print("\n🚀 Starting Species Recognition Training...")
        training_results['species_recognition']['status'] = 'running'
        
        species_success = train_species_recognition()
        training_results['species_recognition']['success'] = species_success
        training_results['species_recognition']['status'] = 'completed' if species_success else 'failed'
        
        if species_success:
            print("✅ Species Recognition training completed successfully!")
        else:
            print("❌ Species Recognition training failed!")
    else:
        print("⏭️  Skipping Species Recognition training")
        training_results['species_recognition']['status'] = 'skipped'
    
    # Training Module B - Health Monitor
    if args.module in ["health", "all"] and not args.skip_health:
        print("\n🚀 Starting Health Monitor Training...")
        training_results['health_monitor']['status'] = 'running'
        
        health_success = train_health_monitor()
        training_results['health_monitor']['success'] = health_success
        training_results['health_monitor']['status'] = 'completed' if health_success else 'failed'
        
        if health_success:
            print("✅ Health Monitor training completed successfully!")
        else:
            print("❌ Health Monitor training failed!")
    else:
        print("⏭️  Skipping Health Monitor training")
        training_results['health_monitor']['status'] = 'skipped'
    
    # Calculate total time
    end_time = time.time()
    total_time = end_time - start_time
    training_results['total_time'] = total_time
    
    # Save training results
    os.makedirs('models', exist_ok=True)
    with open('models/training_results.json', 'w') as f:
        json.dump(training_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Total training time: {total_time/3600:.2f} hours")
    print(f"Species Recognition: {training_results['species_recognition']['status']}")
    print(f"Health Monitor: {training_results['health_monitor']['status']}")
    
    # Success check
    all_success = (
        training_results['species_recognition']['success'] or 
        training_results['species_recognition']['status'] == 'skipped'
    ) and (
        training_results['health_monitor']['success'] or 
        training_results['health_monitor']['status'] == 'skipped'
    )
    
    if all_success:
        print("\n🎉 All training completed successfully!")
        print("Plant AI System is ready for deployment!")
    else:
        print("\n⚠️  Some training modules failed. Check logs for details.")
    
    print(f"\nTraining results saved to: models/training_results.json")
    print(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()







