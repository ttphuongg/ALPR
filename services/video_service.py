import cv2
import time
import threading
import logging
from datetime import datetime
from typing import Callable

import config

logger = logging.getLogger(__name__)


class VideoService:
    def __init__(
        self,
        inference,
        db_handler,
        on_frame_callback: Callable[[object], None],
        on_plate_callback: Callable[[dict], None],
    ) -> None:
        self.inference         = inference
        self.db_handler        = db_handler
        self._on_frame         = on_frame_callback
        self._on_plate         = on_plate_callback

        self._video_path: str | None = None
        self._thread: threading.Thread | None = None
        self._stop_event       = threading.Event()

        # Trạng thái tracking
        self._active_tracks: list[dict] = []
        self._next_track_id: int = 1

        # Anti-Duplicate Layer: {plate_text: last_saved_time}
        self._recent_plates: dict[str, float] = {}

    # Public API 
    def set_video(self, path: str) -> None:
        self._video_path = path
        logger.info("Video path set: %s", path)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            logger.warning("Video service already running.")
            return
        if not self._video_path:
            logger.error("No video path set.")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="VideoServiceThread",
            daemon=True,
        )
        self._thread.start()
        logger.info("Video service thread started.")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("Video service stopped.")

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _calculate_iou(self, boxA: list[int], boxB: list[int]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        interWidth = max(0, xB - xA)
        interHeight = max(0, yB - yA)
        interArea = interWidth * interHeight
        
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        unionArea = boxAArea + boxBArea - interArea
        if unionArea == 0:
            return 0.0
        return interArea / float(unionArea)

    def _vote_plate(self, ocr_results: list[str]) -> str:
        if not ocr_results:
            return ""
        
        # 1. Tìm độ dài chuỗi OCR xuất hiện nhiều nhất trong track (độ dài trội)
        lengths = [len(r) for r in ocr_results]
        length_counts = {}
        for l in lengths:
            length_counts[l] = length_counts.get(l, 0) + 1
        
        dominant_length = max(length_counts, key=length_counts.get)
        
        # 2. Lọc danh sách OCR chỉ giữ lại các chuỗi có độ dài trội đó
        filtered_results = [r for r in ocr_results if len(r) == dominant_length]
        
        # 3. Bỏ phiếu theo từng vị trí ký tự
        voted_chars = []
        for i in range(dominant_length):
            chars_at_pos = [r[i] for r in filtered_results]
            char_counts = {}
            for c in chars_at_pos:
                char_counts[c] = char_counts.get(c, 0) + 1
            best_char = max(char_counts, key=char_counts.get)
            voted_chars.append(best_char)
            
        return "".join(voted_chars)

    def _determine_event(self, plate: str) -> str:
        last_event = self.db_handler.get_last_event(plate)
        return "IN" if (last_event is None or last_event == "OUT") else "OUT"

    def _finish_track(self, track: dict) -> None:
        ocr_list = track["ocr_results"]
        if not ocr_list:
            return
        
        voted_plate = self._vote_plate(ocr_list)
        if not voted_plate:
            return
        
        total_frames = track["total_frames"]
        total_detections = len(ocr_list)
        
        logger.info(
            "Track #%d finished. Total frames=%d, OCR detections=%d, ocr_results=%s, voted=%s",
            track["track_id"], total_frames, total_detections, ocr_list, voted_plate
        )
        
        if total_frames < config.MIN_TRACK_FRAMES or total_detections < config.MIN_OCR_DETECTIONS:
            logger.info(
                "Track #%d discarded (frames %d < %d or detections %d < %d)",
                track["track_id"], total_frames, config.MIN_TRACK_FRAMES,
                total_detections, config.MIN_OCR_DETECTIONS
            )
            return
        
        event_type = self._determine_event(voted_plate)
        detect_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # --- Anti-Duplicate Layer ---
        now = time.time()
        last_saved = self._recent_plates.get(voted_plate)
        if last_saved is not None and (now - last_saved) < config.ANTI_DUPLICATE_SECONDS:
            logger.info(
                "Track #%d DUPLICATE skipped: plate=%s đã ghi %.1f giây trước (ngưỡng=%ds)",
                track["track_id"], voted_plate, now - last_saved, config.ANTI_DUPLICATE_SECONDS,
            )
            return
        # ----------------------------

        # Ghi nhận duy nhất một lần vào database khi kết thúc track
        success = self.db_handler.insert_log(voted_plate, event_type, detect_time)
        if success:
            self._recent_plates[voted_plate] = now
        
        annotated = None
        if track.get("last_frame") is not None and track.get("bbox") is not None:
            annotated = self._draw_bbox(
                track["last_frame"], track["bbox"], voted_plate, event_type
            )
            
        try:
            self._on_plate({
                "track_id":        track["track_id"],
                "plate":           voted_plate,
                "current_ocr":     voted_plate,
                "bbox":            track["bbox"],
                "event_type":      event_type,
                "detect_time":     detect_time,
                "annotated_frame": annotated,
                "is_final":        True,
            })
        except Exception as exc:
            logger.error("on_plate_callback error: %s", exc)

    def _cleanup_tracks(self) -> None:
        now = time.time()
        still_active = []
        
        for track in self._active_tracks:
            if (now - track["last_seen_time"]) > config.TRACK_TIMEOUT:
                self._finish_track(track)
            else:
                still_active.append(track)
                
        self._active_tracks = still_active

    def _draw_bbox(
        self,
        frame,
        bbox: list[int],
        plate_text: str,
        event_type: str,
    ):
        annotated = frame.copy()
        if bbox is None:
            return annotated

        x1, y1, x2, y2 = bbox
        color = (34, 197, 94) if event_type == "IN" else (59, 130, 246)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label = f" {plate_text}  [{event_type}] "
        (lw, lh), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
        )
        label_y1 = max(0, y1 - lh - baseline - 6)
        label_y2 = y1
        cv2.rectangle(
            annotated,
            (x1, label_y1),
            (x1 + lw, label_y2),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            annotated,
            label,
            (x1, max(lh, y1 - baseline - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated

    def _run(self) -> None:
        cap = cv2.VideoCapture(self._video_path)
        if not cap.isOpened():
            logger.error("Cannot open video: %s", self._video_path)
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps > 0 else 25.0
        frame_delay = 1.0 / fps
        frame_count = 0

        logger.info("Video opened. FPS=%.1f, path=%s", fps, self._video_path)

        while not self._stop_event.is_set():
            ret, frame = cap.read()

            if not ret:
                if config.VIDEO_LOOP:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_count = 0
                    logger.debug("Video looped.")
                    continue
                else:
                    logger.info("Video ended, stopping service.")
                    break

            # Cập nhật số frame đếm được cho các track đang active
            for track in self._active_tracks:
                track["total_frames"] += 1

            # Gửi frame tới GUI callback
            try:
                self._on_frame(frame.copy())
            except Exception as exc:
                logger.error("on_frame_callback error: %s", exc)

            # Chạy YOLO mỗi FRAME_SKIP frame
            if frame_count % config.FRAME_SKIP == 0:
                try:
                    result = self.inference.process_frame(frame)
                except Exception as exc:
                    logger.error("Inference error on frame %d: %s", frame_count, exc)
                    result = None

                if result and result.get("plate"):
                    plate_text = result["plate"]
                    
                    # Tìm track khớp nhất bằng IoU
                    best_track = None
                    best_iou = -1.0
                    for track in self._active_tracks:
                        iou = self._calculate_iou(result["bbox"], track["bbox"])
                        if iou > best_iou:
                            best_iou = iou
                            best_track = track
                    
                    if best_track is not None and best_iou >= config.TRACK_IOU_THRESHOLD:
                        best_track["bbox"] = result["bbox"]
                        best_track["ocr_results"].append(plate_text)
                        best_track["last_seen_time"] = time.time()
                        best_track["last_frame"] = frame.copy()
                        track_id = best_track["track_id"]
                        voted_so_far = self._vote_plate(best_track["ocr_results"])
                    else:
                        track_id = self._next_track_id
                        self._next_track_id += 1
                        new_track = {
                            "track_id":        track_id,
                            "bbox":            result["bbox"],
                            "ocr_results":     [plate_text],
                            "start_time":      time.time(),
                            "last_seen_time":  time.time(),
                            "total_frames":    1,
                            "last_frame":      frame.copy(),
                        }
                        self._active_tracks.append(new_track)
                        voted_so_far = plate_text
                    
                    # Gửi cập nhật thời gian thực về GUI (chưa lưu DB)
                    est_event = self._determine_event(voted_so_far)
                    detect_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    annotated = self._draw_bbox(
                        frame, result["bbox"], voted_so_far, est_event
                    )
                    
                    try:
                        self._on_plate({
                            "track_id":        track_id,
                            "plate":           voted_so_far,
                            "current_ocr":     plate_text,
                            "bbox":            result["bbox"],
                            "event_type":      est_event,
                            "detect_time":     detect_time,
                            "annotated_frame": annotated,
                            "is_final":        False,
                        })
                    except Exception as exc:
                        logger.error("on_plate_callback error: %s", exc)

            # Dọn dẹp các track hết hạn
            self._cleanup_tracks()

            frame_count += 1
            time.sleep(frame_delay)

        # Chốt các track còn đang chạy khi dừng hoặc hết video
        for track in self._active_tracks:
            self._finish_track(track)
        self._active_tracks.clear()

        cap.release()
        logger.info("Video capture released.")
