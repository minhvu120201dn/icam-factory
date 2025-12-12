from typing import Optional, Generator
import threading
import cv2
import time

from config import RTSP_SERVER, N_CAMS


class Camera:
    """
    Camera class to handle video capture and frame retrieval.
    """
    last_frame: Optional[cv2.typing.MatLike] = None
    lock = threading.Lock()
    
    def __init__(self, vcap: cv2.VideoCapture):
        """Initialize the Camera with a video capture object.

        Args:
            vcap (cv2.VideoCapture): Video capture object for the camera.
        """
        self.vcap = vcap
        thread = threading.Thread(target=self._update_frame, args=())
        thread.daemon = True
        thread.start()
    
    def _update_frame(self):
        while self.vcap.isOpened():
            with self.lock:
                _, self.last_frame = self.vcap.read()
    
    def get_last_frame(self) -> Optional[cv2.typing.MatLike]:
        """Get the last captured frame.

        Returns:
            Optional[cv2.typing.MatLike]: The last captured frame if available, otherwise None.
        """
        if self.last_frame is not None:
            return self.last_frame.copy()
        else:
            return None
    
    def __del__(self):
        self.vcap.release()


class CameraStreamer:
    """
    Class to stream frames from a Camera instance.
    """
    def __init__(self, cam: Camera):
        self.cam = cam
        self.fps = 0
    
    def __call__(self) -> Generator[cv2.typing.MatLike, None, None]:
        """Stream frames from the camera.

        Yields:
            Generator[cv2.typing.MatLike, None, None]: Frames captured from the camera.
        """
        frame_count = 0
        start_time = time.time()
        
        while True:
            frame = self.cam.get_last_frame()
            if frame is not None:
                yield frame
                frame_count += 1
                
                # Calculate FPS every second
                elapsed_time = time.time() - start_time
                if elapsed_time >= 1.0:
                    self.fps = frame_count / elapsed_time  # Update fps attribute
                    frame_count = 0
                    start_time = time.time()

cams = [
    Camera(cv2.VideoCapture(f"{RTSP_SERVER}/{i}"))
    for i in range(N_CAMS)
]


if __name__ == "__main__":
    camera_streamer = CameraStreamer(cams[0])
    print("Starting frame stream. Press 'q' to quit.")
    
    for frame in camera_streamer():
        # Display FPS on the frame
        fps_text = f"FPS: {camera_streamer.fps:.2f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        # Display the frame
        cv2.imshow("Frame", frame)
        
        # Check for key press ('q' to quit)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        time.sleep(1)  # Simulate ~1 FPS

    cv2.destroyAllWindows()