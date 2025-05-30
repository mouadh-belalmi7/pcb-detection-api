import os
from datetime import timedelta

class Config:
    # Existing configuration
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'static/uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    MODEL_PATH = os.environ.get('MODEL_PATH', 'models/model.tflite')
    
    # Database configuration
    # Use external host for development/testing
    DATABASE_HOST = os.environ.get('DATABASE_HOST', '52.178.110.198')
    DATABASE_USER = os.environ.get('DATABASE_USER', 'guest')
    DATABASE_PASSWORD = os.environ.get('DATABASE_PASSWORD', 'Gue5t_P0stgr3s!_2025')
    DATABASE_NAME = os.environ.get('DATABASE_NAME', 'guest')
    DATABASE_PORT = os.environ.get('DATABASE_PORT', '5432')
    
    # SQLAlchemy configuration
    SQLALCHEMY_DATABASE_URI = f"postgresql://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': 10,
        'pool_recycle': 3600,
        'pool_pre_ping': True
    }

class ProductionConfig(Config):
    # Use internal host when deployed to same network
    DATABASE_HOST = os.environ.get('DATABASE_HOST', 'tkck0kc00s84w4cww0wg0ogc')