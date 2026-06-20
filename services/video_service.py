import cv2
import re
import time
import difflib
import threading
import logging
from datetime import datetime
from typing import Callable

import config
from services.plate_formatter import is_valid_format

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

    @staticmethod
    def _normalize_plate(plate: str) -> str:
        """Bỏ dấu gạch ngang / chấm / khoảng trắng để so sánh thuần ký tự."""
        return re.sub(r"[^A-Z0-9]", "", plate.upper())

    def _finish_track(self, track: dict, early_lock: bool = False) -> None:
        if track.get("is_logged"):
            return
            
        ocr_list = track["ocr_results"]
        if not ocr_list:
            return
        
        voted_plate = track.get("locked_plate") or self._vote_plate(ocr_list)
        if not voted_plate:
            return
        
        total_frames = track["total_frames"]
        total_detections = len(ocr_list)
        
        logger.info(
            "Track #%d %s. frames=%d, OCRs=%d, voted=%s",
            track["track_id"], "early locked" if early_lock else "finished",
            total_frames, total_detections, voted_plate
        )
        
        if not early_lock:
            if total_frames < config.MIN_TRACK_FRAMES or total_detections < config.MIN_OCR_DETECTIONS:
                logger.info("Track #%d discarded", track["track_id"])
                return
        
        event_type = self._determine_event(voted_plate)
        detect_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # --- Anti-Duplicate Layer ---
        now = time.time()
        voted_norm = self._normalize_plate(voted_plate)
        duplicate_of = None
        for saved_plate, saved_time in self._recent_plates.items():
            if (now - saved_time) >= config.ANTI_DUPLICATE_SECONDS:
                continue
            saved_norm = self._normalize_plate(saved_plate)
            ratio = difflib.SequenceMatcher(None, voted_norm, saved_norm).ratio()
            if ratio >= config.PLATE_SIMILARITY_THRESHOLD:
                duplicate_of = (saved_plate, now - saved_time, ratio)
                break

        if duplicate_of is not None:
            logger.info("Track #%d DUPLICATE skipped", track["track_id"])
            track["is_logged"] = True
            return

        success = self.db_handler.insert_log(voted_plate, event_type, detect_time)
        if success:
            self._recent_plates[voted_plate] = now
            
        track["is_logged"] = True
        track["logged_event"] = event_type
        
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
                "just_logged":     success,
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
        return annotated

    def _draw_active_tracks(self, frame) -> "np.ndarray":
        """Vẽ bbox của tất cả track đang active lên frame.

        Sử dụng velocity để dự đoán vị trí bbox hiện tại giữa các YOLO detection,
        giúp bbox trượt mượt theo xe thay vì đứng yên.
        """
        if not self._active_tracks:
            return frame.copy()

        annotated = frame.copy()
        cur_f = self._frame_count

        for track in self._active_tracks:
            bbox = track.get("bbox")
            if bbox is None:
                continue

            # --- Dự đoán vị trí bằng velocity ---
            velocity = track.get("velocity")          # (vx1,vy1,vx2,vy2) pixel/frame
            last_f   = track.get("last_detect_frame", cur_f)
            frames_elapsed = min(cur_f - last_f, config.FRAME_SKIP)  # cap để không bay lê

            if velocity is not None and frames_elapsed > 0:
                vx1, vy1, vx2, vy2 = velocity
                x1 = int(bbox[0] + vx1 * frames_elapsed)
                y1 = int(bbox[1] + vy1 * frames_elapsed)
                x2 = int(bbox[2] + vx2 * frames_elapsed)
                y2 = int(bbox[3] + vy2 * frames_elapsed)
                draw_bbox = [x1, y1, x2, y2]
            else:
                draw_bbox = bbox
            # ------------------------------------

            voted = (
                self._vote_plate(track["ocr_results"])
                if track["ocr_results"]
                else ""
            )
            x1, y1, x2, y2 = draw_bbox
            color = (34, 197, 94)
            if track.get("is_logged"):
                color = (0, 212, 170) # Đổi màu mượt (Accent color) nếu đã khóa
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
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
        self._frame_count = 0  # Dùng bởi _draw_active_tracks

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

            self._frame_count = frame_count

            # Chạy YOLO TRƯỚC khi vẽ → bbox luôn là vị trí mới nhất khi gửi GUI
            if frame_count % config.FRAME_SKIP == 0:
                try:
                    result = self.inference.process_frame(frame)
                except Exception as exc:
                    logger.error("Inference error on frame %d: %s", frame_count, exc)
                    result = None

                if result and result.get("plate"):
                    plate_text = result["plate"]
                    
                    # Tìm track khớp nhất bằng IoU với bbox DỰ ĐOÁN
                    best_track = None
                    best_iou = -1.0
                    for track in self._active_tracks:
                        match_bbox = track["bbox"]
                        velocity = track.get("velocity")
                        last_f = track.get("last_detect_frame", frame_count)
                        frames_elapsed = frame_count - last_f
                        
                        if velocity is not None and frames_elapsed > 0:
                            vx1, vy1, vx2, vy2 = velocity
                            match_bbox = [
                                match_bbox[0] + vx1 * frames_elapsed,
                                match_bbox[1] + vy1 * frames_elapsed,
                                match_bbox[2] + vx2 * frames_elapsed,
                                match_bbox[3] + vy2 * frames_elapsed,
                            ]
                            
                        iou = self._calculate_iou(result["bbox"], match_bbox)
                        if iou > best_iou:
                            best_iou = iou
                            best_track = track
                    
                    if best_track is not None and best_iou >= config.TRACK_IOU_THRESHOLD:
                        old_bbox = best_track["bbox"]
                        new_bbox = result["bbox"]
                        last_f = best_track.get("last_detect_frame", frame_count)
                        elapsed_f = frame_count - last_f
                        if elapsed_f > 0:
                            best_track["velocity"] = [
                                (new_bbox[i] - old_bbox[i]) / elapsed_f
                                for i in range(4)
                            ]
                        best_track["bbox"]              = new_bbox
                        best_track["last_detect_frame"] = frame_count
                        best_track["ocr_results"].append(plate_text)
                        best_track["last_seen_time"]    = time.time()
                        best_track["last_frame"]        = frame.copy()
                        track_id = best_track["track_id"]
                        
                        voted_so_far = best_track.get("locked_plate") or self._vote_plate(best_track["ocr_results"])
                        
                        # Early lock
                        if not best_track.get("is_logged"):
                            if is_valid_format(voted_so_far) and best_track["ocr_results"].count(voted_so_far) >= config.MIN_OCR_DETECTIONS:
                                best_track["locked_plate"] = voted_so_far
                                self._finish_track(best_track, early_lock=True)
                                
                    else:
                        track_id = self._next_track_id
                        self._next_track_id += 1
                        new_track = {
                            "track_id":          track_id,
                            "bbox":              result["bbox"],
                            "ocr_results":       [plate_text],
                            "start_time":        time.time(),
                            "last_seen_time":    time.time(),
                            "total_frames":      1,
                            "last_frame":        frame.copy(),
                            "velocity":          None,
                            "last_detect_frame": frame_count,
                            "is_logged":         False,
                            "locked_plate":      None,
                        }
                        best_track = new_track
                        self._active_tracks.append(new_track)
                        voted_so_far = plate_text
                    
                    is_logged = best_track.get("is_logged", False)
                    if is_logged and "logged_event" in best_track:
                        est_event = best_track["logged_event"]
                    else:
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
                            "is_final":        is_logged,
                            "just_logged":     False,
                        })
                    except Exception as exc:
                        logger.error("on_plate_callback error: %s", exc)

            # Vẽ bbox (đã cập nhật từ YOLO) và gửi tới GUI
            try:
                self._on_frame(self._draw_active_tracks(frame))
            except Exception as exc:
                logger.error("on_frame_callback error: %s", exc)

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
