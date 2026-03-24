# AI-Powered Missing Person Detection System

A real-time web application that uses computer vision and face embeddings to detect and track registered people from a live camera/video stream.

## What this project does

This project combines:

- **Flask** for the web dashboard and API.
- **YOLO (Ultralytics)** for face/person-region detection from frames.
- **InsightFace** for generating facial embeddings.
- **Cosine similarity** to match detected faces against registered targets.
- **OpenCV** for camera streaming, image processing, and snapshot generation.

When a known face is detected above the configured similarity threshold, the system logs the event and stores a snapshot in the alerts folder.

---

## Features

- Start/stop real-time stream from:
  - webcam index (`0`, `1`, etc.),
  - RTSP URL,
  - HTTP video URL.
- Register targets from dashboard (name + image).
- Detect faces in uploaded images via API.
- Live status and FPS in UI.
- Detection logs with timestamp and score.
- Automatic alert snapshots for matched targets.

---

## Project structure

```text
AI-Powered-MPD-System/
├── app.py                  # Flask app + API routes + stream endpoint
├── realtime_detector.py    # Core detection/matching pipeline
├── requirements.txt        # Python dependencies
├── templates/
│   └── dashboard.html      # Web UI
├── static/
│   ├── app.js              # Front-end actions and polling
│   └── styles.css          # Dashboard styling
├── alerts/                 # Saved match snapshots (auto-created)
├── data/
│   └── targets.json        # Registered target embeddings (auto-created)
└── best.pt                 # YOLO model file (you provide this)
```

> Note: `alerts/`, `data/targets.json`, and `best.pt` may not exist in a fresh clone and are created/provided during setup.

---

## Requirements

- Python **3.10+** recommended.
- OS with camera/video support (Linux/Windows/macOS).
- A YOLO model file available at `best.pt` (or custom path via env var).
- Dependencies in `requirements.txt`.

### GPU (optional but recommended)

- CUDA-compatible GPU can improve throughput.
- InsightFace providers default to `CUDAExecutionProvider` then `CPUExecutionProvider` fallback.

---

## How to activate and run the project

### 1) Clone and enter the project

```bash
git clone <your-repo-url>
cd AI-Powered-MPD-System
```

### 2) Create and activate virtual environment

#### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

You should now see `(.venv)` in your terminal prompt.

### 3) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4) Add model file

Place your YOLO model file in project root as:

```text
best.pt
```

Or set `YOLO_MODEL_PATH` to a custom location.

### 5) Start the app

```bash
python app.py
```

Default server:

- URL: `http://127.0.0.1:5000`
- Host: `0.0.0.0`
- Port: `5000` (change with `PORT` env var)

### 6) Use dashboard

Open browser: `http://127.0.0.1:5000`

- Enter source (e.g., `0`) and click **Start Stream**.
- Register person with name + face image.
- Monitor logs and FPS in real time.

---

## Environment variables

The app supports the following variables:

| Variable | Default | Purpose |
|---|---|---|
| `YOLO_MODEL_PATH` | `<project>/best.pt` | Path to YOLO model file |
| `TARGETS_PATH` | `<project>/data/targets.json` | JSON file storing registered embeddings |
| `ALERTS_DIR` | `<project>/alerts` | Folder for saved alert snapshots |
| `FRAME_SKIP` | `2` | Process every Nth frame to balance speed |
| `RESIZE_WIDTH` | `960` | Resize frame width before detection |
| `MATCH_THRESHOLD` | `0.45` | Similarity threshold for known match |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `PORT` | `5000` | Flask server port |

### Example (Linux/macOS)

```bash
export YOLO_MODEL_PATH=/path/to/model.pt
export MATCH_THRESHOLD=0.5
python app.py
```

### Example (Windows PowerShell)

```powershell
$env:YOLO_MODEL_PATH="C:\models\best.pt"
$env:MATCH_THRESHOLD="0.5"
python app.py
```

---

## API endpoints

### `GET /`
Loads dashboard page.

### `POST /api/stream/start`
Start the stream.

**Request JSON:**

```json
{ "source": 0 }
```

`source` can be camera index or URL.

### `POST /api/stream/stop`
Stop active stream.

### `GET /video_feed`
MJPEG stream endpoint consumed by dashboard `<img>` tag.

### `POST /register`
Register target face.

**Form-data fields:**

- `name`: string
- `image`: file

### `POST /detect`
Run detection on uploaded image.

**Form-data fields:**

- `image`: file

### `GET /targets`
List registered target names.

### `GET /api/logs`
Get status, FPS, and recent detection logs.

---

## Common workflow

1. Start app.
2. Start stream (`source=0`).
3. Register one or more targets.
4. Keep stream running and watch logs.
5. Check `alerts/` for saved snapshots when known targets appear.

---

## Troubleshooting

### `YOLO model file not found`
- Ensure `best.pt` exists at project root.
- Or set `YOLO_MODEL_PATH` correctly.

### `InsightFace import failed`
- Reinstall dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Verify Python version compatibility.

### Unable to open source
- Check camera permissions.
- Verify RTSP/HTTP URL is reachable.
- Try another index (`0`, `1`).

### No face detected during registration
- Use a clear frontal face image.
- Increase image quality/lighting.
- Ensure only one prominent face for best results.

### Low FPS
- Increase `FRAME_SKIP` (e.g., 3 or 4).
- Reduce `RESIZE_WIDTH` (e.g., 640).
- Use GPU runtime when available.

---

## Production notes

For production deployment, run with Gunicorn:

```bash
gunicorn -w 1 -k gthread -b 0.0.0.0:5000 app:app
```

(You may tune workers/threads based on hardware and workload.)

---

## Security and privacy notes

- This system processes biometric data (face embeddings).
- Protect `targets.json` and `alerts/` with strict access control.
- Ensure compliance with local privacy and surveillance laws before deployment.

---

## License

Add your preferred license (MIT/Apache-2.0/etc.) in a `LICENSE` file.
