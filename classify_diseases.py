import os

class Config:
    # General Config
    APP_NAME = os.environ.get('APP_NAME', "My Application")
    DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'
    TESTING = os.environ.get('TESTING', 'False').lower() == 'true'
    SECRET_KEY = os.environ.get('SECRET_KEY', 'my_secret_key')
    TIMEZONE = os.environ.get('TIMEZONE', 'UTC')

    # Database Config
    DB_ENGINE = os.environ.get('DB_ENGINE', 'postgresql')  # e.g., postgresql, mysql
    DB_HOST = os.environ.get('DB_HOST', 'localhost')
    DB_PORT = os.environ.get('DB_PORT', 5432)
    DB_NAME = os.environ.get('DB_NAME', 'mydatabase')
    DB_USER = os.environ.get('DB_USER', 'user')
    DB_PASSWORD = os.environ.get('DB_PASSWORD', 'password')
    
    # SQLAlchemy Configuration
    SQLALCHEMY_DATABASE_URI = f"{DB_ENGINE}://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # API Config
    API_VERSION = os.environ.get('API_VERSION', "v1")
    API_BASE_URL = f"/api/{API_VERSION}/"
    API_KEY = os.environ.get('API_KEY', 'my_api_key')
    API_TIMEOUT = int(os.environ.get('API_TIMEOUT', 30))  # seconds

    # Caching Config
    CACHE_TYPE = os.environ.get('CACHE_TYPE', 'simple')  # Can be 'simple', 'redis', etc.
    CACHE_DEFAULT_TIMEOUT = int(os.environ.get('CACHE_DEFAULT_TIMEOUT', 300))
    CACHE_REDIS_URL = os.environ.get('CACHE_REDIS_URL', 'redis://localhost:6379/0')

    # Logging Config
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = os.environ.get('LOG_FILE', 'app.log')
    LOG_MAX_BYTES = int(os.environ.get('LOG_MAX_BYTES', 1024 * 1024 * 10))  # 10 MB
    LOG_BACKUP_COUNT = int(os.environ.get('LOG_BACKUP_COUNT', 5))

    # Email Config
    EMAIL_SERVER = os.environ.get('EMAIL_SERVER', 'smtp.gmail.com')
    EMAIL_PORT = int(os.environ.get('EMAIL_PORT', 587))
    EMAIL_USE_TLS = os.environ.get('EMAIL_USE_TLS', 'true').lower() == 'true'
    EMAIL_USE_SSL = os.environ.get('EMAIL_USE_SSL', 'false').lower() == 'true'
    EMAIL_USERNAME = os.environ.get('EMAIL_USERNAME', 'user@example.com')
    EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD', 'password')
    EMAIL_SENDER = os.environ.get('EMAIL_SENDER', 'no-reply@example.com')

    # Security Config
    SESSION_COOKIE_SECURE = os.environ.get('SESSION_COOKIE_SECURE', 'true').lower() == 'true'
    SESSION_COOKIE_HTTPONLY = os.environ.get('SESSION_COOKIE_HTTPONLY', 'true').lower() == 'true'
    SESSION_COOKIE_SAMESITE = os.environ.get('SESSION_COOKIE_SAMESITE', 'Lax')
    CSRF_ENABLED = os.environ.get('CSRF_ENABLED', 'true').lower() == 'true'
    CSRF_SECRET_KEY = os.environ.get('CSRF_SECRET_KEY', 'my_csrf_secret_key')

    # Rate Limiting Config
    RATE_LIMIT = os.environ.get('RATE_LIMIT', '100 per hour')
    RATE_LIMIT_ENABLED = os.environ.get('RATE_LIMIT_ENABLED', 'false').lower() == 'true'

    # Storage Config
    STORAGE_BACKEND = os.environ.get('STORAGE_BACKEND', 'local')  # Options: 'local', 's3', etc.
    LOCAL_STORAGE_PATH = os.environ.get('LOCAL_STORAGE_PATH', '/path/to/storage/')
    S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME', 'mybucket')
    S3_ACCESS_KEY = os.environ.get('S3_ACCESS_KEY', 'my_access_key')
    S3_SECRET_KEY = os.environ.get('S3_SECRET_KEY', 'my_secret_key')
    S3_REGION_NAME = os.environ.get('S3_REGION_NAME', 'us-east-1')

    # Third-Party API Config
    STRIPE_API_KEY = os.environ.get('STRIPE_API_KEY', 'your_stripe_api_key')
    TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID', 'your_twilio_account_sid')
    TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN', 'your_twilio_auth_token')
    TWILIO_PHONE_NUMBER = os.environ.get('TWILIO_PHONE_NUMBER', '+1234567890')

    # Feature Flags
    FEATURE_FLAG_NEW_UI = os.environ.get('FEATURE_FLAG_NEW_UI', 'false').lower() == 'true'
    FEATURE_FLAG_EXPERIMENTAL = os.environ.get('FEATURE_FLAG_EXPERIMENTAL', 'false').lower() == 'true'
    FEATURE_FLAG_API_V2 = os.environ.get('FEATURE_FLAG_API_V2', 'false').lower() == 'true'

    # File Upload Config
    MAX_UPLOAD_SIZE = int(os.environ.get('MAX_UPLOAD_SIZE', 16 * 1024 * 1024))  # 16 MB
    SUPPORTED_FILE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf'}

    # Session Config
    SESSION_TYPE = os.environ.get('SESSION_TYPE', 'filesystem')  # Options: 'filesystem', 'redis', etc.
    SESSION_PERMANENT = os.environ.get('SESSION_PERMANENT', 'true').lower() == 'true'
    SESSION_USE_SIGNER = os.environ.get('SESSION_USE_SIGNER', 'true').lower() == 'true'

    # Application Timeouts
    REQUEST_TIMEOUT = int(os.environ.get('REQUEST_TIMEOUT', 30))  # seconds
    CONNECTION_TIMEOUT = int(os.environ.get('CONNECTION_TIMEOUT', 30))  # seconds

    # Health Check Config
    HEALTH_CHECK_URL = os.environ.get('HEALTH_CHECK_URL', '/health')
    HEALTH_CHECK_INTERVAL = int(os.environ.get('HEALTH_CHECK_INTERVAL', 60))  # seconds

    # Other Configs
    MAX_CONCURRENT_CONNECTIONS = int(os.environ.get('MAX_CONCURRENT_CONNECTIONS', 100))
    MAX_RETRIES = int(os.environ.get('MAX_RETRIES', 5))
    RETRY_DELAY = int(os.environ.get('RETRY_DELAY', 2))  # seconds

    @staticmethod
    def init_app(app):
        """Initialize the app with configurations."""
        app.config.from_object(Config)

from classify_glaucoma import predict_glaucoma
from classify_diabetes_retinopathy import predict_diabetes_retinopathy
from classify_cataract import predict_cataract
from classify_retinal_detachment import predict_retinal_detachment
from extract_other_fundus import extract_common_features

def predict_disease(img, file):
    predict = None
    if file == 0:
        predict = predict_glaucoma
    elif file == 1:
        predict = predict_diabetes_retinopathy
    elif file == 2:
        predict = predict_cataract
    elif file == 3:
        predict = predict_retinal_detachment
    else:
        predict = extract_common_features
    return predict(img)