from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import uuid
import time
from utils.model_utils import load_tflite_model, process_image, detect_defects_tflite, draw_defect_boxes, print_model_details
import numpy as np
import cv2

# Import database models only if using database
try:
    from flask_migrate import Migrate
    from models import db, Detection, DefectType, DetectionDefect
    from config import Config, ProductionConfig
    USE_DATABASE = True
except ImportError:
    USE_DATABASE = False
    db = None
    
from utils.model_utils import (
    load_tflite_model, process_image, detect_defects_tflite, 
    draw_defect_boxes, print_model_details
)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for Flutter communication

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit file size to 16MB

# Model configuration
MODEL_PATH = 'models/model.tflite'
# Classes for TFLite model - based on the order in your training data
CLASSES = ['mouse_bite', 'spur', 'open_circuit', 'short', 'missing_hole', 'spurious_copper']

# Load model at startup
model = load_tflite_model(MODEL_PATH)
print_model_details(model)

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint to check if the API is running"""
    return jsonify({
        "status": "healthy", 
        "message": "PCB defect detection API is running",
        "model": "TFLite"
    })

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    """Serve uploaded files including result images"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/detect', methods=['POST'])
def detect():
    """Endpoint to detect defects in PCB images"""
    # Check if the post request has the file part
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
        
    file = request.files['image']
    
    # If user does not select file, browser also submits an empty part
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
        
    try:
        # Save the file with a unique name
        extension = os.path.splitext(file.filename)[1]
        if extension.lower() not in ['.jpg', '.jpeg', '.png']:
            return jsonify({'error': 'Unsupported file format. Please upload JPG, JPEG or PNG'}), 400
            
        filename = str(uuid.uuid4()) + extension
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process image and detect defects
        try:
            # Process the image for TFLite model
            processed_image = process_image(filepath)
            results = detect_defects_tflite(model, processed_image, CLASSES)
            
            # Draw bounding boxes on the image if defects were found and we have predictions
            result_image_path = None
            if results.get('defects_found') and 'predictions' in results and results['predictions']:
                result_filename = f"result_{filename}"
                result_filepath = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
                result_image_path = draw_defect_boxes(filepath, results['predictions'], result_filepath)
                if result_image_path:
                    results['result_image'] = os.path.basename(result_image_path)
                
        except Exception as model_error:
            print(f"Model prediction error: {str(model_error)}")
            # Return a more helpful error message
            return jsonify({
                'error': 'Error during image processing or model prediction',
                'details': str(model_error)
            }), 500
            
        # Log the time of detection
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        
        # Generate response
        response = {
            'timestamp': timestamp,
            'success': True,
            'filepath': filepath,
            'image_url': f"/uploads/{filename}",
            'defects': results
        }
        
        # Add result image URL if available
        if result_image_path:
            response['result_image_url'] = f"/uploads/{os.path.basename(result_image_path)}"
        
        return jsonify(response)
        
    except Exception as e:
        print(f"General error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)