import logging
import os
import signal
import time
from pathlib import Path
from typing import Generator

from flask import Flask, Response, jsonify, render_template, request

from realtime_detector import RealtimeDetector


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = os.getenv("YOLO_MODEL_PATH", str(BASE_DIR / "best.pt"))
TARGETS_PATH = os.getenv("TARGETS_PATH", str(BASE_DIR / "data" / "targets.json"))
ALERTS_DIR = os.getenv("ALERTS_DIR", str(BASE_DIR / "alerts"))
FRAME_SKIP = int(os.getenv("FRAME_SKIP", "2"))
RESIZE_WIDTH = int(os.getenv("RESIZE_WIDTH", "960"))
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.45"))


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("mpd-system")

app = Flask(__name__)


def _init_detector() -> RealtimeDetector:
    try:
        detector_instance = RealtimeDetector(
            model_path=MODEL_PATH,
            targets_path=TARGETS_PATH,
            alerts_dir=ALERTS_DIR,
            frame_skip=FRAME_SKIP,
            resize_width=RESIZE_WIDTH,
            threshold=MATCH_THRESHOLD,
        )
        LOGGER.info("Detector initialized successfully")
        return detector_instance
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to initialize detector: %s", exc)
        raise


detector = _init_detector()


@app.route("/")
def dashboard() -> str:
    return render_template("dashboard.html")


@app.route("/api/stream/start", methods=["POST"])
def start_stream():
    payload = request.get_json(silent=True) or {}
    source = payload.get("source", 0)

    try:
        ok, message = detector.start_stream(source)
        return (
            jsonify(
                {
                    "success": ok,
                    "message": message,
                    "source": detector.source,
                    "status": "running" if detector.is_running else "stopped",
                }
            ),
            200 if ok else 409,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Error starting stream: %s", exc)
        return jsonify({"success": False, "message": str(exc)}), 500


@app.route("/api/stream/stop", methods=["POST"])
def stop_stream():
    try:
        ok, message = detector.stop_stream()
        return (
            jsonify({"success": ok, "message": message, "status": "stopped"}),
            200 if ok else 409,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Error stopping stream: %s", exc)
        return jsonify({"success": False, "message": str(exc)}), 500


@app.route("/video_feed")
def video_feed() -> Response:
    return Response(_frame_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


def _frame_generator() -> Generator[bytes, None, None]:
    while True:
        frame = detector.get_jpeg_frame()
        if frame is None:
            time.sleep(0.05)
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"


@app.route("/register", methods=["POST"])
def register_target():
    name = request.form.get("name", "").strip()
    image = request.files.get("image")

    if not name:
        return jsonify({"success": False, "message": "Missing name"}), 400
    if image is None:
        return jsonify({"success": False, "message": "Missing image"}), 400

    try:
        ok, message = detector.register_target(name=name, image_bytes=image.read())
        return jsonify({"success": ok, "message": message}), 200 if ok else 400
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Registration error: %s", exc)
        return jsonify({"success": False, "message": str(exc)}), 500


@app.route("/detect", methods=["POST"])
def detect_image():
    image = request.files.get("image")
    if image is None:
        return jsonify({"success": False, "message": "Missing image", "detections": []}), 400

    try:
        result = detector.detect_image(image.read())
        return jsonify(result), 200 if result["success"] else 400
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Detect error: %s", exc)
        return jsonify({"success": False, "message": str(exc), "detections": []}), 500


@app.route("/targets", methods=["GET"])
def list_targets():
    targets = detector.get_targets()
    return jsonify({"success": True, "count": len(targets), "targets": sorted(targets.keys())})


@app.route("/api/logs", methods=["GET"])
def get_logs():
    return jsonify(
        {
            "success": True,
            "status": "running" if detector.is_running else "stopped",
            "fps": round(detector.fps, 2),
            "logs": detector.get_logs(limit=50),
        }
    )


def _graceful_shutdown(*_args) -> None:
    LOGGER.info("Graceful shutdown requested")
    try:
        detector.stop_stream()
    except Exception:  # noqa: BLE001
        LOGGER.exception("Error during stream shutdown")


signal.signal(signal.SIGINT, _graceful_shutdown)
signal.signal(signal.SIGTERM, _graceful_shutdown)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), threaded=True)
