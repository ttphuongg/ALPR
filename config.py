import os

# Đường dẫn gốc
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model YOLO
MODEL_PLATE_PATH = os.path.join(BASE_DIR, "models", "best.pt")
MODEL_OCR_PATH   = os.path.join(BASE_DIR, "models", "best_ocr.pt")

# Cơ sở dữ liệu
DB_NAME = os.path.join(BASE_DIR, "parking.db")

# Logging
LOG_DIR  = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "app.log")

# Xử lý video
VIDEO_LOOP = False          # Lặp lại video khi kết thúc

# Ngưỡng độ tin cậy
PLATE_CONFIDENCE = 0.5      # Ngưỡng phát hiện biển số
OCR_CONFIDENCE   = 0.4      # Ngưỡng nhận diện ký tự

# YOLO Tracking tích hợp (ByteTrack)
TRACKER = "bytetrack.yaml"  # Tracker Ultralytics – ByteTrack

# Tracking & chốt biển
TRACK_TIMEOUT     = 1.5     # Giây không thấy xe thì kết thúc track
MIN_TRACK_FRAMES  = 5       # Số frame tối thiểu để track hợp lệ
MIN_OCR_DETECTIONS = 5      # Số lần đọc OCR tối thiểu để chốt biển

# Anti-Duplicate Layer (sau Tracking)
ANTI_DUPLICATE_SECONDS    = 30    # Cùng biển số chỉ ghi DB tối đa 1 lần / 30 giây
PLATE_SIMILARITY_THRESHOLD = 0.80 # Fuzzy match: ≥80% ký tự giống → coi là cùng 1 xe

# Giao diện
ANNOTATED_DISPLAY_DURATION = 3    # Giây hiển thị frame có bbox sau khi nhận diện
