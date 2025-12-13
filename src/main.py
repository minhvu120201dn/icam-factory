import cv2
import threading
from typing import List, Callable
from ultralytics import YOLO

from stream import CameraStreamer
from config import RTSP_SERVER, N_CAMS, MODEL_PATH
from detectors import DangerZoneDetector, NoHelmetDetector


model = YOLO(MODEL_PATH)

streamers = [
    CameraStreamer(f"{RTSP_SERVER}/{idx}")
    for idx in range(N_CAMS)
]

# Define danger zones for cameras (example coordinates - adjust as needed)
# Format: List of (x, y) points defining the polygon
DANGER_ZONES = {
    0: [(100, 300), (400, 300), (400, 500), (100, 500)],  # Camera 0
    1: [(150, 250), (450, 250), (450, 480), (150, 480)],  # Camera 1
}

# Initialize detectors for each camera
# Camera 0: both danger zone and helmet detection
# Camera 1: danger zone only
# Camera 2: helmet detection only
camera_detectors = {
    0: {
        "danger_zone": DangerZoneDetector(camera_id=0, zone_polygon=DANGER_ZONES[0]),
        "helmet": NoHelmetDetector(camera_id=0)
    },
    1: {
        "danger_zone": DangerZoneDetector(camera_id=1, zone_polygon=DANGER_ZONES[1])
    },
    2: {
        "helmet": NoHelmetDetector(camera_id=2)
    }
}

def handle_frame(
    frame: cv2.typing.MatLike,
    camera_id: int,
    text: str,
    detectors: dict
):
    """Handle a single frame: run inference and apply detectors.
    
    Args:
        frame: The input frame to process.
        camera_id: The camera ID.
        text: Text to display on frame.
        detectors: Dictionary of detector instances for this camera.
        
    Returns:
        The annotated frame after processing.
    """
    # Run YOLO model with tracking on the frame
    results = model.track(frame, persist=True, verbose=False)
    
    # Get annotated frame with detections and tracking IDs
    annotated_frame = results[0].plot()
    
    # Apply danger zone detector if available
    if "danger_zone" in detectors:
        annotated_frame = detectors["danger_zone"].check_zone(results, annotated_frame)
    
    # Apply helmet detector if available
    if "helmet" in detectors:
        annotated_frame = detectors["helmet"].check_helmet(results, annotated_frame)
    
    # Show info as text
    if text:
        cv2.putText(annotated_frame, text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    return annotated_frame

_existed_windows = set()

def process_stream(
    camera_idx: int,
    streamer: CameraStreamer,
    stop_event: threading.Event,
):
    """Process a single camera stream in a separate thread.
    
    Args:
        camera_idx (int): The index of the camera stream.
        streamer (CameraStreamer): The CameraStreamer instance.
        stop_event (threading.Event): Event to signal stopping the thread.
    """
    # Ensure unique window names
    global _existed_windows
    window_name = f"Camera {camera_idx}"
    assert window_name not in _existed_windows, f"Window {window_name} already exists!"
    _existed_windows.add(window_name)
    
    # Get detectors for this camera
    detectors = camera_detectors.get(camera_idx, {})
    
    # Log which detectors are active
    active_detectors = []
    if "danger_zone" in detectors:
        active_detectors.append("Danger Zone")
    if "helmet" in detectors:
        active_detectors.append("Helmet Detection")
    
    print(f"Camera {camera_idx} active detectors: {', '.join(active_detectors) if active_detectors else 'None'}")
    
    # Stream frames from the camera
    for frame in streamer():
        # Check if stop event is set
        if stop_event.is_set():
            break
        
        # Handle the frame with appropriate detectors
        annotated_frame = handle_frame(
            frame,
            camera_idx,
            f"Camera {camera_idx} | FPS: {streamer.fps:.2f}",
            detectors
        )
        
        # Display the annotated frame
        cv2.imshow(window_name, annotated_frame)
        
        # Check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break
    
    cv2.destroyWindow(window_name)

def main():
    print(f"Starting inference on {N_CAMS} camera streams...")
    print("Press 'q' to quit.")
    
    stop_event = threading.Event()
    
    # Start a thread for each streamer
    threads = []
    for idx, streamer in enumerate(streamers):
        thread = threading.Thread(target=process_stream, args=(idx, streamer, stop_event))
        thread.daemon = True
        thread.start()
        threads.append(thread)
    
    print(f"Started {len(threads)} camera threads")
    
    # Wait for all threads to finish
    for thread in threads:
        thread.join()
    
    print("All streams stopped")

if __name__ == "__main__":
    main()