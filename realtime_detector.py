import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO

try:
    from insightface.app import FaceAnalysis
except Exception as exc:  # noqa: BLE001
    FaceAnalysis = None
    _INSIGHTFACE_IMPORT_ERROR = exc
else:
    _INSIGHTFACE_IMPORT_ERROR = None


LOGGER = logging.getLogger(__name__)


@dataclass
class DetectionEvent:
    name: str
    timestamp: str
    snapshot_path: str
    score: float


class RealtimeDetector:
    def __init__(
        self,
        model_path: str,
        targets_path: str,
        alerts_dir: str,
        frame_skip: int = 2,
        resize_width: int = 960,
        threshold: float = 0.45,
    ) -> None:
        self.model_path = Path(model_path)
        self.targets_path = Path(targets_path)
        self.alerts_dir = Path(alerts_dir)
        self.alerts_dir.mkdir(parents=True, exist_ok=True)

        self.frame_skip = max(1, frame_skip)
        self.resize_width = max(320, resize_width)
        self.threshold = threshold

        self._stream_lock = threading.Lock()
        self._data_lock = threading.Lock()

        self._running = False
        self._source: Union[int, str] = 0
        self._capture: Optional[cv2.VideoCapture] = None

        self._latest_jpeg: Optional[bytes] = None
        self._latest_boxes: List[Dict] = []
        self._fps: float = 0.0
        self._frame_count = 0
        self._worker: Optional[threading.Thread] = None

        self._targets: Dict[str, np.ndarray] = {}
        self._logs: List[DetectionEvent] = []
        self._last_alert_by_name: Dict[str, float] = {}

        self._model = None
        self._face_analyzer = None

        self._load_models()
        self.load_targets()

    def _load_models(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO model file not found: {self.model_path}")

        LOGGER.info("Loading YOLO model from %s", self.model_path)
        self._model = YOLO(str(self.model_path))

        if FaceAnalysis is None:
            raise RuntimeError(f"InsightFace import failed: {_INSIGHTFACE_IMPORT_ERROR}")

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        LOGGER.info("Initializing InsightFace with providers: %s", providers)
        self._face_analyzer = FaceAnalysis(providers=providers)
        self._face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def source(self) -> Union[int, str]:
        return self._source

    def _parse_source(self, source: Union[int, str]) -> Union[int, str]:
        if isinstance(source, int):
            return source
        if isinstance(source, str) and source.strip().isdigit():
            return int(source.strip())
        return source

    def _ensure_capture(self, source: Union[int, str]) -> cv2.VideoCapture:
        parsed_source = self._parse_source(source)
        capture = cv2.VideoCapture(parsed_source)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Unable to open video source: {parsed_source}")
        return capture

    def start_stream(self, source: Union[int, str] = 0) -> Tuple[bool, str]:
        with self._stream_lock:
            if self._running:
                return False, "Stream already running"

            self._capture = self._ensure_capture(source)
            self._source = self._parse_source(source)
            self._running = True
            self._worker = threading.Thread(target=self._process_loop, daemon=True)
            self._worker.start()
            LOGGER.info("Stream started for source: %s", self._source)
            return True, "Stream started"

    def stop_stream(self) -> Tuple[bool, str]:
        with self._stream_lock:
            if not self._running:
                return False, "Stream is not running"

            self._running = False
            worker = self._worker

        if worker and worker.is_alive():
            worker.join(timeout=2)

        with self._stream_lock:
            if self._capture is not None:
                self._capture.release()
                self._capture = None
            self._worker = None
            LOGGER.info("Stream stopped")

        return True, "Stream stopped"

    def _resize_for_detection(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if w <= self.resize_width:
            return frame
        ratio = self.resize_width / w
        new_size = (self.resize_width, int(h * ratio))
        return cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)

    def _extract_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        results = self._model(frame, verbose=False)
        boxes: List[Tuple[int, int, int, int]] = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1i, y1i, x2i, y2i = map(lambda v: int(max(0, v)), (x1, y1, x2, y2))
                if (x2i - x1i) > 20 and (y2i - y1i) > 20:
                    boxes.append((x1i, y1i, x2i, y2i))

        return boxes

    def _embedding_from_crop(self, face_bgr: np.ndarray) -> Optional[np.ndarray]:
        if face_bgr.size == 0:
            return None

        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        faces = self._face_analyzer.get(rgb)
        if not faces:
            return None

        embedding = faces[0].embedding
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return None
        return embedding / norm

    def _find_match(self, embedding: np.ndarray) -> Tuple[str, float]:
        with self._data_lock:
            if not self._targets:
                return "Unknown", 0.0

            names = list(self._targets.keys())
            matrix = np.vstack([self._targets[name] for name in names])

        sims = cosine_similarity([embedding], matrix)[0]
        best_idx = int(np.argmax(sims))
        score = float(sims[best_idx])
        if score >= self.threshold:
            return names[best_idx], score
        return "Unknown", score

    def _draw_box(self, frame: np.ndarray, box: Tuple[int, int, int, int], label: str, score: float) -> None:
        x1, y1, x2, y2 = box
        is_match = label != "Unknown"
        color = (0, 0, 255) if is_match else (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {score:.2f}" if score > 0 else label
        cv2.putText(frame, text, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def _save_alert(self, frame: np.ndarray, name: str, score: float) -> None:
        now = datetime.utcnow()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S UTC")
        throttle_key = name
        epoch_now = time.time()

        last_seen = self._last_alert_by_name.get(throttle_key, 0)
        if epoch_now - last_seen < 3:
            return

        filename = f"{name}_{now.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        path = self.alerts_dir / filename
        cv2.imwrite(str(path), frame)

        event = DetectionEvent(name=name, timestamp=timestamp, snapshot_path=str(path), score=score)
        with self._data_lock:
            self._logs.insert(0, event)
            self._logs = self._logs[:200]
            self._last_alert_by_name[throttle_key] = epoch_now

    def _process_loop(self) -> None:
        frame_idx = 0
        last_ts = time.time()

        while self._running:
            capture = self._capture
            if capture is None:
                break

            ok, frame = capture.read()
            if not ok:
                LOGGER.warning("Failed to read frame from source %s", self._source)
                time.sleep(0.05)
                continue

            frame_idx += 1
            processed = self._resize_for_detection(frame.copy())
            boxes_drawn: List[Dict] = []

            if frame_idx % self.frame_skip == 0:
                boxes = self._extract_faces(processed)

                for box in boxes:
                    x1, y1, x2, y2 = box
                    crop = processed[y1:y2, x1:x2]
                    embedding = self._embedding_from_crop(crop)

                    if embedding is None:
                        label, score = "Unknown", 0.0
                    else:
                        label, score = self._find_match(embedding)
                        if label != "Unknown":
                            self._save_alert(processed, label, score)

                    boxes_drawn.append(
                        {
                            "label": label,
                            "score": round(score, 4),
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                        }
                    )
                    self._draw_box(processed, box, label, score)

            self._frame_count += 1
            now = time.time()
            elapsed = now - last_ts
            if elapsed >= 1.0:
                self._fps = self._frame_count / elapsed
                self._frame_count = 0
                last_ts = now

            cv2.putText(
                processed,
                f"FPS: {self._fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2,
            )

            ok, jpeg = cv2.imencode(".jpg", processed)
            if ok:
                self._latest_jpeg = jpeg.tobytes()
                self._latest_boxes = boxes_drawn

        LOGGER.info("Exiting frame processing loop")

    def get_jpeg_frame(self) -> Optional[bytes]:
        return self._latest_jpeg

    def get_logs(self, limit: int = 30) -> List[Dict]:
        with self._data_lock:
            return [event.__dict__ for event in self._logs[:limit]]

    def load_targets(self) -> None:
        if not self.targets_path.exists():
            self.targets_path.parent.mkdir(parents=True, exist_ok=True)
            self.targets_path.write_text("{}", encoding="utf-8")
            self._targets = {}
            return

        raw = json.loads(self.targets_path.read_text(encoding="utf-8"))
        loaded = {}
        for name, emb in raw.items():
            vec = np.array(emb, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                loaded[name] = vec / norm
        self._targets = loaded

    def save_targets(self) -> None:
        with self._data_lock:
            payload = {name: embedding.tolist() for name, embedding in self._targets.items()}
        self.targets_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def register_target(self, name: str, image_bytes: bytes) -> Tuple[bool, str]:
        np_img = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if image is None:
            return False, "Invalid image"

        resized = self._resize_for_detection(image)
        boxes = self._extract_faces(resized)
        if not boxes:
            return False, "No face detected"

        x1, y1, x2, y2 = boxes[0]
        crop = resized[y1:y2, x1:x2]
        embedding = self._embedding_from_crop(crop)
        if embedding is None:
            return False, "Unable to generate embedding"

        with self._data_lock:
            self._targets[name] = embedding
        self.save_targets()
        return True, f"Target '{name}' registered"

    def detect_image(self, image_bytes: bytes) -> Dict:
        np_img = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if image is None:
            return {"success": False, "message": "Invalid image", "detections": []}

        resized = self._resize_for_detection(image)
        boxes = self._extract_faces(resized)
        detections = []

        for box in boxes:
            x1, y1, x2, y2 = box
            crop = resized[y1:y2, x1:x2]
            embedding = self._embedding_from_crop(crop)
            if embedding is None:
                name, score = "Unknown", 0.0
            else:
                name, score = self._find_match(embedding)

            detections.append(
                {
                    "name": name,
                    "score": round(score, 4),
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                }
            )

        return {"success": True, "message": "Detection complete", "detections": detections}

    def get_targets(self) -> Dict[str, List[float]]:
        with self._data_lock:
            return {name: embedding.tolist() for name, embedding in self._targets.items()}
