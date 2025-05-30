from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_migrate import Migrate
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
import os
import uuid
import time
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

from config import Config, ProductionConfig
from models import db, Detection, DefectType, DetectionDefect
from utils.model_utils import (
    load_tflite_model, process_image, detect_defects_tflite, 
    draw_defect_boxes, print_model_details
)

# Initialize Flask app
app = Flask(__name__)

# Load configuration
env = os.environ.get('FLASK_ENV', 'development')
if env == 'production':
    app.config.from_object(ProductionConfig)
else:
    app.config.from_object(Config)

# Initialize extensions
CORS(app, origins=app.config.get('CORS_ORIGINS', '*').split(','))
db.init_app(app)
migrate = Migrate(app, db)

# Setup ProxyFix
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

# Configure logging
if not app.debug:
    if not os.path.exists('logs'):
        os.mkdir('logs')
    file_handler = RotatingFileHandler('logs/pcb_detection.log', maxBytes=10240000, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('PCB Detection API startup')

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Model configuration
CLASSES = ['mouse_bite', 'spur', 'open_circuit', 'short', 'missing_hole', 'spurious_copper']
model = None

# Initialize database and model
with app.app_context():
    try:
        # Create tables if they don't exist
        db.create_all()
        
        # Initialize defect types if not exists
        for class_name in CLASSES:
            defect_type = DefectType.query.filter_by(name=class_name).first()
            if not defect_type:
                colors = {
                    'missing_hole': '#FF0000',      # Red
                    'mouse_bite': '#00FF00',        # Green
                    'open_circuit': '#0000FF',      # Blue
                    'short': '#FFFF00',             # Yellow
                    'spur': '#FF00FF',              # Magenta
                    'spurious_copper': '#00FFFF'    # Cyan
                }
                defect_type = DefectType(
                    name=class_name,
                    color_hex=colors.get(class_name, '#FFFFFF')
                )
                db.session.add(defect_type)
        db.session.commit()
        app.logger.info('Database initialized successfully')
    except Exception as e:
        app.logger.error(f'Database initialization error: {str(e)}')

# Load model
try:
    model = load_tflite_model(app.config['MODEL_PATH'])
    print_model_details(model)
    app.logger.info('Model loaded successfully')
except Exception as e:
    app.logger.error(f'Failed to load model: {str(e)}')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with database status"""
    db_status = "connected"
    try:
        # Test database connection
        db.session.execute(db.text('SELECT 1'))
        detection_count = Detection.query.count()
    except Exception as e:
        db_status = f"error: {str(e)}"
        detection_count = 0
    
    return jsonify({
        "status": "healthy",
        "message": "PCB defect detection API is running",
        "model": "TFLite",
        "model_status": "loaded" if model else "not loaded",
        "database_status": db_status,
        "total_detections": detection_count,
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route('/detect', methods=['POST'])
def detect():
    """Main detection endpoint with database storage"""
    start_time = time.time()
    
    if not model:
        return jsonify({'error': 'Model not loaded'}), 503
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Save uploaded file
        extension = os.path.splitext(file.filename)[1]
        filename = str(uuid.uuid4()) + extension
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get image dimensions
        from PIL import Image
        with Image.open(filepath) as img:
            img_width, img_height = img.size
        
        # Process image
        processed_image = process_image(filepath)
        results = detect_defects_tflite(model, processed_image, CLASSES)
        
        # Create detection record
        detection = Detection(
            original_image_path=f"/uploads/{filename}",
            defects_found=results.get('defects_found', False),
            total_defects=len(results.get('predictions', [])),
            image_width=img_width,
            image_height=img_height,
            client_ip=request.remote_addr,
            user_agent=request.headers.get('User-Agent', '')[:255]
        )
        
        # Draw bounding boxes if defects found
        if results.get('defects_found') and results.get('predictions'):
            result_filename = f"result_{filename}"
            result_filepath = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
            result_image_path = draw_defect_boxes(filepath, results['predictions'], result_filepath)
            
            if result_image_path:
                detection.result_image_path = f"/uploads/{result_filename}"
                results['result_image'] = result_filename
            
            # Store individual defects
            for pred in results['predictions']:
                defect = DetectionDefect(
                    defect_type=pred['defect_type'],
                    confidence=pred['confidence'],
                    x_min=pred['location'][0],
                    y_min=pred['location'][1],
                    x_max=pred['location'][2],
                    y_max=pred['location'][3]
                )
                detection.defects.append(defect)
                
                # Update defect type statistics
                defect_type = DefectType.query.filter_by(name=pred['defect_type']).first()
                if defect_type:
                    defect_type.total_count += 1
        
        # Set processing time and predictions
        processing_time = time.time() - start_time
        detection.processing_time = processing_time
        detection.predictions = results.get('predictions', [])
        
        # Save to database
        db.session.add(detection)
        db.session.commit()
        
        app.logger.info(
            f'Detection completed - ID: {detection.id}, '
            f'Defects: {detection.defects_found}, '
            f'Count: {detection.total_defects}, '
            f'Time: {processing_time:.3f}s'
        )
        
        # Build response
        response = {
            'success': True,
            'detection_id': detection.id,
            'timestamp': detection.timestamp.isoformat(),
            'processing_time': round(processing_time, 3),
            'image_url': f"/uploads/{filename}",
            'defects': results
        }
        
        if detection.result_image_path:
            response['result_image_url'] = detection.result_image_path
        
        return jsonify(response)
        
    except Exception as e:
        app.logger.error(f'Detection error: {str(e)}', exc_info=True)
        db.session.rollback()
        return jsonify({
            'error': 'An error occurred during processing',
            'message': str(e) if app.debug else 'Internal server error'
        }), 500

@app.route('/history', methods=['GET'])
def get_history():
    """Get detection history with pagination"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        
        # Query with pagination
        pagination = Detection.query.order_by(
            Detection.timestamp.desc()
        ).paginate(
            page=page, 
            per_page=per_page, 
            error_out=False
        )
        
        return jsonify({
            'success': True,
            'detections': [d.to_dict() for d in pagination.items],
            'total': pagination.total,
            'page': page,
            'per_page': per_page,
            'total_pages': pagination.pages
        })
    except Exception as e:
        app.logger.error(f'History error: {str(e)}')
        return jsonify({'error': 'Failed to fetch history'}), 500

@app.route('/statistics', methods=['GET'])
def get_statistics():
    """Get detection statistics"""
    try:
        # Overall statistics
        total_detections = Detection.query.count()
        total_with_defects = Detection.query.filter_by(defects_found=True).count()
        
        # Defect type statistics
        defect_types = DefectType.query.all()
        
        # Recent detections (last 24 hours)
        from datetime import datetime, timedelta
        since = datetime.utcnow() - timedelta(hours=24)
        recent_count = Detection.query.filter(
            Detection.timestamp >= since
        ).count()
        
        # Average processing time
        avg_time = db.session.query(
            db.func.avg(Detection.processing_time)
        ).scalar() or 0
        
        return jsonify({
            'success': True,
            'statistics': {
                'total_detections': total_detections,
                'detections_with_defects': total_with_defects,
                'detection_rate': round(total_with_defects / total_detections * 100, 2) if total_detections > 0 else 0,
                'recent_detections_24h': recent_count,
                'average_processing_time': round(avg_time, 3),
                'defect_types': [d.to_dict() for d in defect_types]
            }
        })
    except Exception as e:
        app.logger.error(f'Statistics error: {str(e)}')
        return jsonify({'error': 'Failed to fetch statistics'}), 500

@app.route('/detection/<int:detection_id>', methods=['GET'])
def get_detection(detection_id):
    """Get specific detection details"""
    try:
        detection = Detection.query.get_or_404(detection_id)
        return jsonify({
            'success': True,
            'detection': detection.to_dict()
        })
    except Exception as e:
        return jsonify({'error': 'Detection not found'}), 404

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    """Serve uploaded files"""
    try:
        filename = secure_filename(filename)
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)