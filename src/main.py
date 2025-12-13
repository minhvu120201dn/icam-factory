import cv2
import threading
from queue import Queue
from ultralytics import YOLO

from stream import CameraStreamer
from config import RTSP_SERVER, N_CAMS, MODEL_PATH
from detectors import DangerZoneDetector, NoHelmetDetector


streamers = [
    CameraStreamer(f"{RTSP_SERVER}/{idx}")
    for idx in range(N_CAMS)
]

# Define danger zones for cameras (example coordinates - adjust as needed)
# Format: List of (x, y) points defining the polygon
DANGER_ZONES = {
    0: [(559, 175), (766, 73), (766, 156), (704, 201)],  # Camera 0
    1: [(216, 224), (441, 272), (527, 267), (563, 288), (356, 283), (121, 227)],  # Camera 1
    2: [(232, 362), (415, 355), (765, 417), (764, 458), (669, 473)],  # Camera 2
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
    text: str,
    detectors: dict,
    model: YOLO
):
    """Handle a single frame: run inference and apply detectors.
    
    Args:
        frame: The input frame to process.
        text: Text to display on frame.
        detectors: Dictionary of detector instances for this camera.
        model: YOLO model instance for this thread.
        
    Returns:
        The annotated frame after processing.
    """
    # Run YOLO model with tracking on the frame
    results = model.track(frame, persist=True, verbose=False, conf=0.5)
    
    # Get annotated frame with detections and tracking IDs
    annotated_frame = results[0].plot()
    
    # Apply danger zone detector if available
    if "danger_zone" in detectors:
        annotated_frame = detectors["danger_zone"].check_zone(results, annotated_frame)
    
    # Apply helmet detector if available
    if "helmet" in detectors:
        annotated_frame = detectors["helmet"].check_helmet(results, annotated_frame)
    
    # Show info as textwebm
    if text:
        cv2.putText(annotated_frame, text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    return annotated_frame

def process_stream(
    camera_idx: int,
    streamer: CameraStreamer,
    stop_event: threading.Event,
    frame_queue: Queue,
):
    """Process a single camera stream in a separate thread.
    
    Args:
        camera_idx (int): The index of the camera stream.
        streamer (CameraStreamer): The CameraStreamer instance.
        stop_event (threading.Event): Event to signal stopping the thread.
        frame_queue (Queue): Queue to send processed frames to main thread.
    """
    # Create a separate YOLO model instance for this thread
    model = YOLO(MODEL_PATH)
    
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
            f"Camera {camera_idx} | FPS: {streamer.fps:.2f}",
            detectors,
            model
        )
        
        # Send the processed frame to the main thread for display
        try:
            frame_queue.put((camera_idx, annotated_frame), block=False)
        except:
            pass  # Queue full, skip this frame

def main():
    print(f"Starting inference on {N_CAMS} camera streams...")
    print("Press 'q' to quit.")
    
    stop_event = threading.Event()
    frame_queue = Queue(maxsize=N_CAMS * 2)
    
    # Start a thread for each streamer
    threads = []
    for idx, streamer in enumerate(streamers):
        # Create and start thread
        thread = threading.Thread(target=process_stream, args=(idx, streamer, stop_event, frame_queue))
        thread.daemon = True
        thread.start()
        threads.append(thread)
    
    print(f"Started {len(threads)} camera threads")
    
    # Main thread handles display (OpenCV requires this for thread safety)
    try:
        while not stop_event.is_set():
            # Get processed frames from queue
            if not frame_queue.empty():
                camera_idx, annotated_frame = frame_queue.get()
                window_name = f"Camera {camera_idx}"
                cv2.imshow(window_name, annotated_frame)
            
            # Check for 'q' key to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                stop_event.set()
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        stop_event.set()
    
    finally:
        # Wait for all threads to finish
        for thread in threads:
            thread.join(timeout=2.0)
        
        cv2.destroyAllWindows()
        print("All streams stopped")

if __name__ == "__main__":
    main()