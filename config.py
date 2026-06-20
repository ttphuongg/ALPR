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
FRAME_SKIP = 3            # Chạy YOLO mỗi N frame
VIDEO_LOOP = False         # Lặp lại video khi kết thúc

# gưỡng độ tin cậy 
PLATE_CONFIDENCE = 0.5    # Ngưỡng phát hiện biển số
OCR_CONFIDENCE   = 0.4    # Ngưỡng nhận diện ký tự

# Chống lưu trùng & Tracking
DUPLICATE_TIMEOUT = 5     # Giây tối thiểu giữa 2 lần ghi cùng biển số
TRACK_TIMEOUT = 1.5       # Giây không thấy xe thì kết thúc track
TRACK_IOU_THRESHOLD = 0.2 # Ngưỡng IoU đè lên nhau để xác định cùng 1 xe (nới lỏng cho biển số nhỏ)
MIN_TRACK_FRAMES = 5      # Số frame tối thiểu để hợp lệ
MIN_OCR_DETECTIONS = 3    # Số lần đọc OCR tối thiểu trong track

# Anti-Duplicate Layer (sau Tracking)
ANTI_DUPLICATE_SECONDS = 30      # Cùng biển số chỉ được ghi DB tối đa 1 lần / 30 giây
PLATE_SIMILARITY_THRESHOLD = 0.80  # Fuzzy match: ≥80% ký tự giống → coi là cùng 1 xe

# Giao diện 
ANNOTATED_DISPLAY_DURATION = 3   # Giây hiển thị frame có bbox sau khi nhận diện
