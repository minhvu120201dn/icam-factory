from typing import Literal, List
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

def process_stream(
    idx: int,
    streamer: CameraStreamer,
    stop_event: threading.Event,
):
    """Process a single camera stream in a separate thread.
    
    Args:
        idx (int): The index of the camera stream.
        streamer (CameraStreamer): The CameraStreamer instance.
        stop_event (threading.Event): Event to signal stopping the thread.
    """    
    for frame in streamer():
        # Check if stop event is set
        if stop_event.is_set():
            break
        
        # Run YOLO model on the frame
        results = model(frame, verbose=False)
        
        # Get annotated frame with detections
        annotated_frame = results[0].plot()
        
        # Add camera ID and FPS info
        fps_text = f"Camera {idx} | FPS: {streamer.fps:.2f}"
        cv2.putText(annotated_frame, fps_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Display the annotated frame
        cv2.imshow(f"Camera {idx}", annotated_frame)
        
        # Check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break
    
    cv2.destroyWindow(f"Camera {idx}")

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