import os
import sys
import logging

import config

os.makedirs(config.LOG_DIR, exist_ok=True)
os.makedirs(os.path.join(config.BASE_DIR, "videos"), exist_ok=True)

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(config.LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)
logger.info("=" * 60)
logger.info("Smart Parking Management System – Starting up")
logger.info("BASE_DIR : %s", config.BASE_DIR)
logger.info("DB_NAME  : %s", config.DB_NAME)
logger.info("LOG_FILE : %s", config.LOG_FILE)
logger.info("=" * 60)


# Kiểm tra model files
def _check_models() -> bool:
    ok = True
    for name, path in [
        ("Plate detection (best.pt)",    config.MODEL_PLATE_PATH),
        ("OCR characters (best_ocr.pt)", config.MODEL_OCR_PATH),
    ]:
        if os.path.isfile(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            logger.info("✓ %-35s → %s  (%.1f MB)", name, path, size_mb)
        else:
            logger.error("✗ Model không tìm thấy: %s", path)
            ok = False
    return ok


# Khởi động ứng dụng
def main() -> None:
    if not _check_models():
        logger.error(
            "Một hoặc nhiều model files không tìm thấy. "
            "Vui lòng kiểm tra thư mục models/ và đảm bảo các file "
            "best.pt và best_ocr.pt đã được đặt đúng chỗ."
        )


    # Khởi tạo database
    from database.db_handler import DBHandler
    try:
        db = DBHandler()
        logger.info("Database handler initialized.")
    except Exception as exc:
        logger.critical("Cannot initialize database: %s", exc)
        sys.exit(1)

    # Khởi tạo và chạy GUI
    from gui.app import ParkingApp
    try:
        app = ParkingApp()
        app.set_db(db)
        logger.info("GUI launched. Entering mainloop.")
        app.mainloop()
    except Exception as exc:
        logger.critical("Fatal error in GUI: %s", exc, exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Application exited.")


if __name__ == "__main__":
    main()
