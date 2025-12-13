# iCAM Factory - Construction Site Safety Monitoring System

Real-time AI-powered safety monitoring system for construction sites using YOLO object detection and tracking. Detects safety violations including:
- Workers entering dangerous zones
- Workers without helmets
- Real-time alerts with automatic snapshot capture

## 🎯 Features

- **Multi-camera support**: Monitor up to 3 cameras simultaneously
- **Danger zone detection**: Define custom polygonal danger zones via interactive UI
- **Helmet detection**: Identify workers without proper safety equipment
- **Object tracking**: Persistent tracking across frames with unique IDs
- **Alert system**: Automatic alerts with snapshot capture to SQLite database
- **RTSP streaming**: Stream video files as RTSP for testing/development
- **Thread-safe architecture**: Parallel processing with main thread rendering

## 📋 Requirements

- Python 3.13+
- FFmpeg (for RTSP streaming)
- MediaMTX (RTSP server)
- CUDA-capable GPU (recommended)

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/minhvu120201dn/icam-factory.git
cd icam-factory
```

### 2. Create a virtual environment

```bash
conda create -n icam-factory python=3.13
conda activate icam-factory
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Install MediaMTX (RTSP Server)

**Option A: Download standalone binary**
```bash
wget https://github.com/bluenviron/mediamtx/releases/download/v1.5.0/mediamtx_v1.5.0_linux_amd64.tar.gz
tar -xzf mediamtx_v1.5.0_linux_amd64.tar.gz
```

**Option B: Use Docker**
```bash
docker run --rm -it -p 8554:8554 bluenviron/mediamtx
```

### 5. Set up environment variables

Create a `.env` file in the project root:

```bash
RTSP_SERVER=rtsp://localhost:8554
N_CAMS=3
MODEL_PATH=pretrained/helmet-head-person.pt
```

## 📚 Training the Model (Optional)

If you want to train your own model:

### 1. Prepare your dataset

Place your dataset in `datasets/` following YOLO format:
```
datasets/
  YourDataset/
    data.yaml
    train/
      images/
      labels/
    valid/
      images/
      labels/
```

### 2. Configure training

Edit the dataset path in your training script or notebook (see `notebooks/1.Head-and-Helmet.ipynb`).

### 3. Train the model

```python
from ultralytics import YOLO

# Load a pretrained model
model = YOLO('yolo11n.pt')

# Train the model
results = model.train(
    data='datasets/YourDataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='helmet-detection'
)
```

### 4. Export the trained model

The trained model will be saved in `runs/detect/helmet-detection/weights/best.pt`. Copy it to `pretrained/` and update your `.env` file.

## 🎬 Running the System

### Step 1: Start MediaMTX RTSP Server

In a separate terminal:

```bash
./mediamtx
```

Or with Docker:
```bash
docker run --rm -it -p 8554:8554 bluenviron/mediamtx
```

### Step 2: Stream video files as RTSP (for testing)

Configure your video sources in the Makefile or use environment variables:

```bash
# Edit input videos
export INPUT_VIDEO_0=datasets/TownCentre/input_video_s2_l1_01.mp4
export INPUT_VIDEO_1=datasets/TownCentre/input_video_s2_l1_03.mp4
export INPUT_VIDEO_2=datasets/TownCentre/input_video_s2_l1_04.mp4

# Start all 3 RTSP streams
make rtsp_servers
```

Or start individual streams:
```bash
make rtsp_server_0
make rtsp_server_1
make rtsp_server_2
```

The streams will be available at:
- `rtsp://localhost:8554/0`
- `rtsp://localhost:8554/1`
- `rtsp://localhost:8554/2`

### Step 3: Define danger zones (first time setup)

Run the interactive zone definition tool:

```bash
cd src
python zone_definer.py
```

**Instructions:**
- Choose which cameras to define zones for
- Left-click to add polygon points
- Right-click to remove the last point
- Press SPACE to capture a new frame if needed
- Press ENTER when done (minimum 3 points)
- Press ESC to cancel

The tool will:
- Save zones to `danger_zones.json`
- Generate Python code to copy into `main.py`

Copy the generated `DANGER_ZONES` dictionary into `src/main.py`.

### Step 4: Run the monitoring system

```bash
cd src
python main.py
```

The system will:
- Connect to all 3 RTSP streams
- Process frames in parallel threads
- Display annotated video with detections
- Trigger alerts for safety violations
- Save snapshots to `snapshots/`
- Log alerts to `alerts.db`

**Controls:**
- Press `q` to quit

## 🏗️ Project Structure

```
icam-factory/
├── datasets/               # Training datasets
├── pretrained/            # Pre-trained model weights
├── runs/                  # Training outputs
├── snapshots/            # Alert snapshots (auto-generated)
├── notebooks/            # Jupyter notebooks for experiments
├── src/
│   ├── main.py           # Main monitoring application
│   ├── stream.py         # Camera streaming classes
│   ├── config.py         # Configuration management
│   ├── detectors.py      # Danger zone & helmet detectors
│   ├── alerts.py         # Alert system & database
│   └── zone_definer.py   # Interactive zone definition tool
├── Makefile              # RTSP server management
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables
├── alerts.db            # SQLite alerts database (auto-generated)
└── danger_zones.json    # Danger zone definitions (auto-generated)
```

## 🔧 Configuration

### Camera Configuration

Edit `src/config.py` or `.env`:

```python
RTSP_SERVER = "rtsp://localhost:8554"  # RTSP server URL
N_CAMS = 3                              # Number of cameras
MODEL_PATH = "pretrained/helmet-head-person.pt"  # Model path
```

### Detector Configuration

Edit `src/main.py`:

```python
camera_detectors = {
    0: {  # Camera 0: both detectors
        "danger_zone": DangerZoneDetector(...),
        "helmet": NoHelmetDetector(...)
    },
    1: {  # Camera 1: danger zone only
        "danger_zone": DangerZoneDetector(...)
    },
    2: {  # Camera 2: helmet only
        "helmet": NoHelmetDetector(...)
    }
}
```

### RTSP Streaming Configuration

Edit `Makefile` to change video sources:

```makefile
INPUT_VIDEO_0 ?= datasets/video1.mp4
INPUT_VIDEO_1 ?= datasets/video2.mp4
INPUT_VIDEO_2 ?= datasets/video3.mp4
RTSP_BASE_URL ?= rtsp://localhost:8554
```

## 📊 Alert System

Alerts are automatically saved to `alerts.db` with:
- Timestamp
- Camera ID
- Alert type (danger_zone / no_helmet)
- Track ID of the detected person
- Snapshot image path
- Additional details

### Query alerts programmatically:

```python
from alerts import get_recent_alerts

# Get last 10 alerts from all cameras
alerts = get_recent_alerts(limit=10)

# Get last 10 alerts from camera 0
alerts = get_recent_alerts(camera_id=0, limit=10)

for alert in alerts:
    print(f"{alert['timestamp']}: {alert['alert_type']} on Camera {alert['camera_id']}")
```

## 🎥 Using with Real Cameras

To use with real RTSP cameras instead of video files:

1. Update `.env` or `config.py` with your camera URLs:

```python
streamers = [
    CameraStreamer("rtsp://camera1.local/stream"),
    CameraStreamer("rtsp://camera2.local/stream"),
    CameraStreamer("rtsp://camera3.local/stream"),
]
```

2. Skip Step 2 (RTSP streaming) and run directly:

```bash
python src/main.py
```

## 🐛 Troubleshooting

### "Could not open RTSP stream"
- Ensure MediaMTX is running
- Check that video files exist at specified paths
- Verify RTSP URLs are correct

### Threading warnings
- The system uses thread-safe architecture with main thread handling display
- Qt warnings can be safely ignored

### Low FPS
- Reduce inference resolution: `imgsz=320` in model.track()
- Use a smaller model: `yolo11n.pt` instead of `yolo11x.pt`
- Enable GPU: Install CUDA and PyTorch with CUDA support

### Model loading errors
- Ensure model path is correct in `.env`
- Check model file is not corrupted
- Try re-downloading the model weights

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or support, please open an issue on GitHub.

## 🙏 Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - Object detection framework
- [MediaMTX](https://github.com/bluenviron/mediamtx) - RTSP server
- [OpenCV](https://opencv.org/) - Computer vision library
- [Hard Hat Workers Dataset](https://universe.roboflow.com/roboflow-universe-projects/hard-hat-workers) - Roboflow Universe dataset for helmet detection training
- [Oxford Town Centre Dataset](https://www.kaggle.com/datasets/ashayajbani/oxford-town-centre) - Oxford Active Vision Lab pedestrian tracking dataset, used for the demo
