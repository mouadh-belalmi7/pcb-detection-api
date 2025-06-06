from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

db = SQLAlchemy()

class Detection(db.Model):
    """Model for storing detection results"""
    __tablename__ = 'detections'
    
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    
    # Image information
    original_image_path = db.Column(db.String(255), nullable=False)
    result_image_path = db.Column(db.String(255))
    image_width = db.Column(db.Integer)
    image_height = db.Column(db.Integer)
    
    # Detection results
    defects_found = db.Column(db.Boolean, default=False)
    total_defects = db.Column(db.Integer, default=0)
    processing_time = db.Column(db.Float)  # in seconds
    
    # Store predictions as JSON
    predictions_json = db.Column(db.Text)
    
    # Optional metadata
    client_ip = db.Column(db.String(45))  # IPv4 or IPv6
    user_agent = db.Column(db.String(255))
    
    # Add indexes for common queries
    __table_args__ = (
        db.Index('idx_timestamp', 'timestamp'),
        db.Index('idx_defects_found', 'defects_found'),
    )
    
    @property
    def predictions(self):
        """Get predictions as Python object"""
        if self.predictions_json:
            return json.loads(self.predictions_json)
        return []
    
    @predictions.setter
    def predictions(self, value):
        """Set predictions from Python object"""
        self.predictions_json = json.dumps(value)
    
    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'original_image': self.original_image_path,
            'result_image': self.result_image_path,
            'defects_found': self.defects_found,
            'total_defects': self.total_defects,
            'processing_time': self.processing_time,
            'predictions': self.predictions,
            'image_dimensions': {
                'width': self.image_width,
                'height': self.image_height
            }
        }

class DefectType(db.Model):
    """Model for storing defect types and statistics"""
    __tablename__ = 'defect_types'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    description = db.Column(db.Text)
    color_hex = db.Column(db.String(7))  # For visualization
    total_count = db.Column(db.Integer, default=0)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'color': self.color_hex,
            'total_count': self.total_count
        }

class DetectionDefect(db.Model):
    """Association table for detection-defect relationship"""
    __tablename__ = 'detection_defects'
    
    id = db.Column(db.Integer, primary_key=True)
    detection_id = db.Column(db.Integer, db.ForeignKey('detections.id'), nullable=False)
    defect_type = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    x_min = db.Column(db.Float, nullable=False)
    y_min = db.Column(db.Float, nullable=False)
    x_max = db.Column(db.Float, nullable=False)
    y_max = db.Column(db.Float, nullable=False)
    
    # Relationships
    detection = db.relationship('Detection', backref='defects')
