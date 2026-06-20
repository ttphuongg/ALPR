import cv2
import numpy as np
import logging
from ultralytics import YOLO

import config
from services.plate_formatter import format_vietnamese_plate

logger = logging.getLogger(__name__)


class PlateInference:
    def __init__(self) -> None:
        # Load model phát hiện biển số
        try:
            self.plate_model: YOLO = YOLO(config.MODEL_PLATE_PATH)
            logger.info("Plate detection model loaded: %s", config.MODEL_PLATE_PATH)
        except Exception as exc:
            logger.error("Failed to load plate model from '%s': %s",
                         config.MODEL_PLATE_PATH, exc)
            raise

        # Load model OCR ký tự 
        try:
            self.ocr_model: YOLO = YOLO(config.MODEL_OCR_PATH)
            logger.info("OCR model loaded: %s", config.MODEL_OCR_PATH)
            logger.info("OCR class names: %s", self.ocr_model.names)
        except Exception as exc:
            logger.error("Failed to load OCR model from '%s': %s",
                         config.MODEL_OCR_PATH, exc)
            raise

    # Phát hiện biển số

    def detect_plate(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray | None, list[int] | None]:
        if frame is None or frame.size == 0:
            return None, None

        try:
            results = self.plate_model(
                frame,
                conf=config.PLATE_CONFIDENCE,
                verbose=False,
            )

            if not results or len(results[0].boxes) == 0:
                return None, None

            boxes = results[0].boxes

            # Chọn detection có confidence cao nhất
            best_idx = int(boxes.conf.argmax())
            xyxy = boxes.xyxy[best_idx].cpu().numpy()
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

            # Giới hạn và mở rộng bbox
            h, w = frame.shape[:2]
            pad_x = int(0.05 * (x2 - x1))
            pad_y = int(0.05 * (y2 - y1))

            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)

            if x2 <= x1 or y2 <= y1:
                logger.debug("Invalid bbox after clamping: [%d,%d,%d,%d]", x1, y1, x2, y2)
                return None, None

            plate_crop = frame[y1:y2, x1:x2].copy()
            logger.debug("Plate detected at [%d,%d,%d,%d]", x1, y1, x2, y2)
            return plate_crop, [x1, y1, x2, y2]

        except Exception as exc:
            logger.error("detect_plate error: %s", exc)
            return None, None

    # OCR ký tự biển số 

    def ocr_plate(self, plate_crop: np.ndarray, is_two_lines: bool) -> list[str]:
        if plate_crop is None or plate_crop.size == 0:
            return []

        try:
            results = self.ocr_model(
                plate_crop,
                conf=config.OCR_CONFIDENCE,
                verbose=False,
            )

            if not results or len(results[0].boxes) == 0:
                return []

            boxes = results[0].boxes
            char_list: list[tuple[float, float, str]] = []

            for i in range(len(boxes)):
                cls_id    = int(boxes.cls[i].item())
                x_center  = float(boxes.xywh[i][0].item())
                y_center  = float(boxes.xywh[i][1].item())
                char      = self.ocr_model.names[cls_id]
                char_list.append((x_center, y_center, char))

            if not is_two_lines:
                char_list.sort(key=lambda c: c[0])
                plate_text = "".join(c[2] for c in char_list)
                logger.debug("OCR result 1-line: '%s'", plate_text)
                return [plate_text]
            else:
                if not char_list:
                    return []
                y_mean = sum(c[1] for c in char_list) / len(char_list)
                top_line = [c for c in char_list if c[1] < y_mean]
                bottom_line = [c for c in char_list if c[1] >= y_mean]

                top_line.sort(key=lambda c: c[0])
                bottom_line.sort(key=lambda c: c[0])

                str_top = "".join(c[2] for c in top_line)
                str_bottom = "".join(c[2] for c in bottom_line)
                logger.debug("OCR result 2-lines: top='%s', bottom='%s'", str_top, str_bottom)
                return [str_top, str_bottom]

        except Exception as exc:
            logger.error("ocr_plate error: %s", exc)
            return []

    def process_frame(self, frame: np.ndarray) -> dict | None:
        plate_crop, bbox = self.detect_plate(frame)

        if plate_crop is None:
            return None

        # Phát hiện biển 2 dòng dựa vào aspect ratio của vùng crop
        # Biển 2 dòng thường cao hơn rộng (tỷ lệ h/w > 0.5)
        h_crop, w_crop = plate_crop.shape[:2]
        is_two_lines = (h_crop / w_crop) > 0.5 if w_crop > 0 else False

        raw_chars_list = self.ocr_plate(plate_crop, is_two_lines)

        if not raw_chars_list or all(len(line) == 0 for line in raw_chars_list):
            return None

        # Format chuẩn biển số Việt Nam
        formatted = format_vietnamese_plate(raw_chars_list, is_two_lines)
        if not formatted:
            return None

        flat_plate = "".join(raw_chars_list)
        logger.debug(
            "plate raw=%s → formatted='%s' (two_lines=%s)",
            raw_chars_list, formatted, is_two_lines,
        )

        return {
            "plate": formatted,
            "bbox":  bbox,
        }
