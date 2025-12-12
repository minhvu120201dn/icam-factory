import cv2
import threading
from ultralytics import YOLO

from stream import CameraStreamer
from config import RTSP_SERVER, N_CAMS, MODEL_PATH


model = YOLO(MODEL_PATH)

streamers = [
    CameraStreamer(f"{RTSP_SERVER}/{idx}")
    for idx in range(N_CAMS)
]

def handle_frame(frame: cv2.typing.MatLike, text: str):
    """Handle a single frame: run inference and display results.
    
    Args:
        frame: The input frame to process.
        idx (int): The index of the camera stream.
        streamer (CameraStreamer): The CameraStreamer instance.
        
    Returns:
        The annotated frame after processing.
    """
    # Run YOLO model with tracking on the frame
    results = model.track(frame, persist=True, verbose=False)
    
    # Get annotated frame with detections and tracking IDs
    annotated_frame = results[0].plot()
    
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
    global _existed_windows
    window_name = f"Camera {camera_idx}"
    assert window_name not in _existed_windows, f"Window {window_name} already exists!"
    _existed_windows.add(window_name)
    
    for frame in streamer():
        # Check if stop event is set
        if stop_event.is_set():
            break
        
        # Handle the frame
        annotated_frame = handle_frame(frame, f"Camera {camera_idx} | FPS: {streamer.fps:.2f}")
        
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