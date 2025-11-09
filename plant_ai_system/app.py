"""
Plant AI Web Application
Giao diện web để upload ảnh và nhận diện sâu bệnh
"""

import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI API integration
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    if OPENAI_API_KEY:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        print("OpenAI API initialized successfully")
    else:
        openai_client = None
        print("Warning: OPENAI_API_KEY not found in environment variables")
        OPENAI_AVAILABLE = False
except ImportError:
    OPENAI_AVAILABLE = False
    openai_client = None
    print("Warning: OpenAI package not installed. Install with: pip install openai")
except Exception as e:
    OPENAI_AVAILABLE = False
    openai_client = None
    print(f"Warning: OpenAI initialization failed: {e}")

# Load disease database
DISEASE_DB_PATH = os.path.join(os.path.dirname(__file__), 'disease_database.json')
try:
    with open(DISEASE_DB_PATH, 'r', encoding='utf-8') as f:
        DISEASE_DATABASE = json.load(f)
    print(f"Disease database loaded successfully: {len(DISEASE_DATABASE.get('diseases', {}))} diseases")
except FileNotFoundError:
    print(f"Warning: Disease database not found at {DISEASE_DB_PATH}")
    DISEASE_DATABASE = {'diseases': {}, 'plant_types': [], 'total_diseases': 0}
except Exception as e:
    print(f"Error loading disease database: {e}")
    DISEASE_DATABASE = {'diseases': {}, 'plant_types': [], 'total_diseases': 0}

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
app.secret_key = 'plant_ai_secret_key_2024'

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static/results', exist_ok=True)


class PlantAIModel:
    """Plant AI Model for web interface"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.classes = None
        self.transform = None
        self.load_model()
    
    def load_model(self):
        """Load trained model"""
        model_path = "models/best_plant_model.pth"
        
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
            
            print(f"Model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict_image(self, image_path):
        """Predict plant disease from image"""
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
            
            # Health analysis
            health_analysis = self.analyze_plant_health(predicted_class, confidence_score)
            
            # Get OpenAI treatment advice if available and treatment is needed
            openai_advice = None
            if OPENAI_AVAILABLE and openai_client and health_analysis.get('treatment_needed', False):
                openai_advice = get_openai_treatment_advice(
                    disease_name=health_analysis.get('disease_name', predicted_class),
                    plant_type=health_analysis.get('plant_type', 'Unknown'),
                    symptoms=health_analysis.get('symptoms', []),
                    severity=health_analysis.get('severity', 'Unknown')
                )
            
            return {
                'success': True,
                'predicted_class': predicted_class,
                'confidence': confidence_score,
                'top_predictions': top_predictions,
                'health_analysis': health_analysis,
                'openai_advice': openai_advice,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def analyze_plant_health(self, predicted_class, confidence):
        """Analyze plant health based on prediction using comprehensive database"""
        # Try to find exact match in database
        if predicted_class in DISEASE_DATABASE['diseases']:
            disease_info = DISEASE_DATABASE['diseases'][predicted_class]
            return {
                'status': disease_info['status'],
                'score': disease_info['health_score'],
                'severity': disease_info['severity'],
                'recommendations': disease_info['treatment']['recommendations'],
                'treatment_needed': disease_info['treatment']['needed'],
                'description': disease_info.get('description', ''),
                'symptoms': disease_info.get('symptoms', []),
                'prevention': disease_info.get('prevention', []),
                'plant_type': disease_info.get('plant_type', 'Unknown'),
                'disease_name': disease_info.get('name', predicted_class)
            }
        
        # Try partial matching for common patterns
        predicted_lower = predicted_class.lower()
        
        if 'healthy' in predicted_lower:
            # Determine plant type
            plant_type = 'Plant'
            for pt in DISEASE_DATABASE['plant_types']:
                if pt.lower() in predicted_lower:
                    plant_type = pt
                    break
            
            return {
                'status': 'Healthy',
                'score': 95,
                'severity': 'None',
                'recommendations': [
                    f'{plant_type} is healthy and thriving',
                    'Continue current care routine',
                    'Monitor regularly for any changes',
                    'Apply preventive measures in appropriate seasons'
                ],
                'treatment_needed': False,
                'description': f'The {plant_type.lower()} is in excellent health with no signs of disease.',
                'plant_type': plant_type,
                'disease_name': f'Healthy {plant_type}'
            }
        else:
            # Try to find similar disease
            for disease_key, disease_info in DISEASE_DATABASE['diseases'].items():
                if disease_key.lower() in predicted_lower or predicted_lower in disease_key.lower():
                    return {
                        'status': disease_info['status'],
                        'score': disease_info['health_score'],
                        'severity': disease_info['severity'],
                        'recommendations': disease_info['treatment']['recommendations'],
                        'treatment_needed': disease_info['treatment']['needed'],
                        'description': disease_info.get('description', ''),
                        'symptoms': disease_info.get('symptoms', []),
                        'prevention': disease_info.get('prevention', []),
                        'plant_type': disease_info.get('plant_type', 'Unknown'),
                        'disease_name': disease_info.get('name', predicted_class)
                    }
            
            # Default for unknown conditions
            return {
                'status': 'Unknown Condition',
                'score': 70,
                'severity': 'Unknown',
                'recommendations': [
                    'Consult with plant expert',
                    'Monitor plant closely',
                    'Take additional photos from different angles',
                    'Check environmental conditions',
                    'Compare with known disease symptoms'
                ],
                'treatment_needed': False,
                'description': 'Unable to identify specific condition. Please consult an expert.',
                'plant_type': 'Unknown',
                'disease_name': predicted_class
            }


# Initialize model
plant_ai = PlantAIModel()


def get_openai_treatment_advice(disease_name, plant_type, symptoms=None, severity=None):
    """
    Get detailed treatment advice from OpenAI API
    
    Args:
        disease_name: Name of the disease
        plant_type: Type of plant
        symptoms: List of symptoms (optional)
        severity: Severity level (optional)
    
    Returns:
        Dictionary with treatment advice, medicines, and care instructions
    """
    if not OPENAI_AVAILABLE or not openai_client:
        return None
    
    try:
        symptoms_text = ", ".join(symptoms) if symptoms else "Not specified"
        severity_text = severity if severity else "Unknown"
        
        prompt = f"""You are an expert plant pathologist and agricultural consultant. Provide detailed treatment advice for the following plant disease:

Plant Type: {plant_type}
Disease: {disease_name}
Severity: {severity_text}
Symptoms: {symptoms_text}

Please provide a comprehensive response in JSON format with the following structure:
{{
    "overview": "Brief overview of the disease",
    "treatment_steps": [
        "Step 1 description",
        "Step 2 description"
    ],
    "medicines": [
        {{
            "name": "Medicine name",
            "type": "Fungicide/Insecticide/Bactericide/etc",
            "application": "How to apply",
            "dosage": "Recommended dosage",
            "frequency": "How often to apply",
            "safety": "Safety precautions"
        }}
    ],
    "care_instructions": [
        "Care instruction 1",
        "Care instruction 2"
    ],
    "prevention": [
        "Prevention tip 1",
        "Prevention tip 2"
    ],
    "timeline": "Expected recovery timeline",
    "monitoring": "What to monitor during treatment"
}}

Be specific, practical, and provide actionable advice. Focus on organic and chemical treatment options. Return only valid JSON."""

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",  # Using gpt-4o-mini for cost efficiency
            messages=[
                {"role": "system", "content": "You are an expert plant pathologist providing detailed treatment advice. Always respond in valid JSON format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        
        # Parse JSON response
        content = response.choices[0].message.content.strip()
        
        # Try to extract JSON if wrapped in markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        advice = json.loads(content)
        return advice
        
    except json.JSONDecodeError as e:
        print(f"Error parsing OpenAI JSON response: {e}")
        print(f"Response content: {content[:200]}")
        return None
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and prediction"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    if file and allowed_file(file.filename):
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Predict
            result = plant_ai.predict_image(filepath)
            
            if result['success']:
                # Save result
                result['uploaded_file'] = filename
                result['file_path'] = filepath
                
                # Save to results
                result_file = f"static/results/result_{timestamp}.json"
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                return jsonify(result)
            else:
                return jsonify(result)
                
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    return jsonify({'success': False, 'error': 'Invalid file type'})


@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for prediction"""
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'success': False, 'error': 'No image data provided'})
        
        # Decode base64 image
        image_data = data['image'].split(',')[1]  # Remove data:image/jpeg;base64,
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        # Save temporary image
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_path = f"static/uploads/temp_{timestamp}.jpg"
        image.save(temp_path)
        
        # Predict
        result = plant_ai.predict_image(temp_path)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/history')
def history():
    """Show prediction history"""
    results_dir = 'static/results'
    if not os.path.exists(results_dir):
        return render_template('history.html', results=[])
    
    results = []
    for filename in os.listdir(results_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(results_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                    results.append(result)
            except:
                continue
    
    # Sort by timestamp
    results.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    
    return render_template('history.html', results=results)


@app.route('/about')
def about():
    """About page"""
    stats = {
        'total_plant_types': len(DISEASE_DATABASE['plant_types']),
        'total_diseases': DISEASE_DATABASE['total_diseases'],
        'plant_types': DISEASE_DATABASE['plant_types']
    }
    return render_template('about.html', stats=stats)

@app.route('/api/stats')
def get_stats():
    """API endpoint for statistics"""
    return jsonify({
        'total_plant_types': len(DISEASE_DATABASE['plant_types']),
        'total_diseases': DISEASE_DATABASE['total_diseases'],
        'plant_types': DISEASE_DATABASE['plant_types'],
        'model_loaded': plant_ai.model is not None,
        'total_classes': len(plant_ai.classes) if plant_ai.classes else 0
    })


if __name__ == '__main__':
    print("Plant AI Web Application")
    print("=" * 40)
    print(f"Model loaded: {plant_ai.model is not None}")
    print(f"Classes: {len(plant_ai.classes) if plant_ai.classes else 0}")
    print(f"Device: {plant_ai.device}")
    print("\nStarting web server...")
    print("Open your browser and go to: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)




