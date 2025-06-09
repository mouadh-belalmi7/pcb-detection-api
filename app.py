
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
import os
import uuid
import time
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

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

# Simple configuration for Render
class SimpleConfig:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key")
    UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "static/uploads")
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
    MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.tflite")
    CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")

# Load configuration
if USE_DATABASE:
    env = os.environ.get("FLASK_ENV", "development")
    if env == "production":
        app.config.from_object(ProductionConfig)
    else:
        app.config.from_object(Config)
else:
    app.config.from_object(SimpleConfig)

# Initialize CORS
CORS(app, origins=app.config.get("CORS_ORIGINS", ["*"]))

# Initialize database if available
if USE_DATABASE:
    db.init_app(app)
    migrate = Migrate(app, db)

# Setup ProxyFix
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

# Configure logging
if not app.debug:
    if not os.path.exists("logs"):
        os.mkdir("logs")
    file_handler = RotatingFileHandler("logs/pcb_detection.log", maxBytes=10240000, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s"
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info("PCB Detection API startup")

# Create upload folder
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Model configuration
CLASSES = ["mouse_bite", "spur", "open_circuit", "short", "missing_hole", "spurious_copper"]
model = None
model_loaded = False

# Lazy load model to avoid memory issues on startup
def load_model_lazy():
    global model, model_loaded
    if not model_loaded:
        try:
            model_path = app.config.get("MODEL_PATH", "models/model.tflite")
            
            # Debug: Check if file exists and get file info
            app.logger.info(f"Looking for model at: {model_path}")
            app.logger.info(f"Current working directory: {os.getcwd()}")
            app.logger.info(f"Files in current directory: {os.listdir('.')}")
            
            if os.path.exists("models"):
                app.logger.info(f"Files in models directory: {os.listdir('models')}")
                # Check file size
                if os.path.exists(model_path):
                    file_size = os.path.getsize(model_path)
                    app.logger.info(f"Model file exists, size: {file_size} bytes")
                    if file_size < 1000:  # If file is too small, it might be a Git LFS pointer
                        app.logger.error(f"Model file seems too small ({file_size} bytes). This might be a Git LFS pointer file.")
                        with open(model_path, 'r') as f:
                            content = f.read(200)  # Read first 200 characters
                            app.logger.error(f"File content preview: {content}")
                else:
                    app.logger.error(f"Model file not found at {model_path}")
            else:
                app.logger.error("models directory does not exist")
            
            if os.path.exists(model_path):
                app.logger.info(f"Loading model from {model_path}...")
                model = load_tflite_model(model_path)
                model_loaded = True
                app.logger.info("Model loaded successfully")
                try:
                    print_model_details(model)
                except Exception as e:
                    app.logger.warning(f"Could not print model details: {e}")
            else:
                app.logger.error(f"Model file not found at {model_path}")
        except Exception as e:
            app.logger.error(f"Failed to load model: {str(e)}")
            import traceback
            app.logger.error(traceback.format_exc())
    return model

# Initialize database tables if using database
if USE_DATABASE:
    with app.app_context():
        try:
            db.create_all()
            app.logger.info("Database tables created")
        except Exception as e:
            app.logger.error(f"Database initialization error: {str(e)}")

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

@app.route("/")
def index():
    """Root endpoint"""
    return jsonify({
        "name": "PCB Detection API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "detect": "/detect (POST)",
            "history": "/history" if USE_DATABASE else None,
            "statistics": "/statistics" if USE_DATABASE else None
        }
    })

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    health_data = {
        "status": "healthy",
        "message": "PCB defect detection API is running",
        "model": "TFLite",
        "model_status": "loaded" if model_loaded else "not loaded yet",
        "database": "enabled" if USE_DATABASE else "disabled",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if USE_DATABASE:
        try:
            db.session.execute(db.text("SELECT 1"))
            health_data["database_status"] = "connected"
        except Exception as e:
            health_data["database_status"] = f"error: {str(e)}"
    
    return jsonify(health_data)

@app.route("/detect", methods=["POST"])
def detect():
    """Main detection endpoint"""
    start_time = time.time()
    
    # Load model on first request
    current_model = load_model_lazy()
    if not current_model:
        return jsonify({"error": "Model not available. Please check server logs."}), 503
    
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No image selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Allowed types: jpg, jpeg, png"}), 400
    
    try:
        # Save uploaded file
        extension = os.path.splitext(file.filename)[1]
        filename = str(uuid.uuid4()) + extension
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        app.logger.info(f"Saved uploaded file: {filename}")
        
        # Get image dimensions
        img_width, img_height = None, None
        try:
            from PIL import Image
            with Image.open(filepath) as img:
                img_width, img_height = img.size
        except Exception as e:
            app.logger.warning(f"Could not get image dimensions: {e}")
        
        # Process image
        processed_image = process_image(filepath)
        results = detect_defects_tflite(current_model, processed_image, CLASSES)
        
        # Draw bounding boxes if defects found
        result_image_url = None
        if results.get("defects_found") and results.get("predictions"):
            result_filename = f"result_{filename}"
            result_filepath = os.path.join(app.config["UPLOAD_FOLDER"], result_filename)
            result_image_path = draw_defect_boxes(filepath, results["predictions"], result_filepath)
            
            if result_image_path:
                result_image_url = f"/uploads/{result_filename}"
                results["result_image"] = result_filename
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Save to database if enabled
        detection_id = None
        if USE_DATABASE:
            try:
                detection = Detection(
                    original_image_path=f"/uploads/{filename}",
                    result_image_path=result_image_url,
                    defects_found=results.get("defects_found", False),
                    total_defects=len(results.get("predictions", [])),
                    image_width=img_width,
                    image_height=img_height,
                    processing_time=processing_time,
                    client_ip=request.remote_addr,
                    user_agent=str(request.headers.get("User-Agent", ""))[:255]
                )
                detection.predictions = results.get("predictions", [])
                
                db.session.add(detection)
                db.session.commit()
                detection_id = detection.id
                app.logger.info(f"Saved detection to database with ID: {detection_id}")
            except Exception as e:
                app.logger.error(f"Database save error: {e}")
                db.session.rollback()
        
        # Build response
        response = {
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "processing_time": round(processing_time, 3),
            "image_url": f"/uploads/{filename}",
            "defects": results
        }
        
        if detection_id:
            response["detection_id"] = detection_id
        
        if result_image_url:
            response["result_image_url"] = result_image_url
        
        app.logger.info(f"Detection completed in {processing_time:.3f}s")
        return jsonify(response)
        
    except Exception as e:
        app.logger.error(f"Detection error: {str(e)}", exc_info=True)
        return jsonify({
            "error": "An error occurred during processing",
            "message": str(e) if app.debug else "Internal server error"
        }), 500

@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    """Serve uploaded files"""
    try:
        filename = secure_filename(filename)
        return send_from_directory(app.config["UPLOAD_FOLDER"], filename)
    except Exception as e:
        app.logger.error(f"File not found: {filename}")
        return jsonify({"error": "File not found"}), 404

# Database endpoints (only if database is enabled)
if USE_DATABASE:
    @app.route("/history", methods=["GET"])
    def get_history():
        """Get detection history with pagination"""
        try:
            page = request.args.get("page", 1, type=int)
            per_page = request.args.get("per_page", 20, type=int)
            
            pagination = Detection.query.order_by(
                Detection.timestamp.desc()
            ).paginate(
                page=page, 
                per_page=per_page, 
                error_out=False
            )
            
            return jsonify({
                "success": True,
                "detections": [d.to_dict() for d in pagination.items],
                "total": pagination.total,
                "page": page,
                "per_page": per_page,
                "total_pages": pagination.pages
            })
        except Exception as e:
            app.logger.error(f"History error: {str(e)}")
            return jsonify({"error": "Failed to fetch history"}), 500

    @app.route("/statistics", methods=["GET"])
    def get_statistics():
        """Get detection statistics"""
        try:
            total_detections = Detection.query.count()
            total_with_defects = Detection.query.filter_by(defects_found=True).count()
            
            return jsonify({
                "success": True,
                "statistics": {
                    "total_detections": total_detections,
                    "detections_with_defects": total_with_defects,
                    "detection_rate": round(total_with_defects / total_detections * 100, 2) if total_detections > 0 else 0
                }
            })
        except Exception as e:
            app.logger.error(f"Statistics error: {str(e)}")
            return jsonify({"error": "Failed to fetch statistics"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)


