from typing import Optional, Generator
import threading
import cv2

from config import RTSP_SERVER, N_CAMS


class Camera:
    last_frame: Optional[cv2.typing.MatLike] = None
    lock = threading.Lock()
    
    def __init__(self, vcap: cv2.VideoCapture):
        self.vcap = vcap
        thread = threading.Thread(target=self.update, args=())
        thread.daemon = True
        thread.start()
    
    def update(self):
        while self.vcap.isOpened():
            with self.lock:
                _, self.last_frame = self.vcap.read()
    
    def get_last_frame(self) -> Optional[cv2.typing.MatLike]:
        if self.last_frame is not None:
            return self.last_frame.copy()
        else:
            return None
    
    def __del__(self):
        self.vcap.release()

    
def stream_last_frame(cam: Camera) -> Generator[cv2.typing.MatLike, None, None]:
    while True:
        frame = cam.get_last_frame()
        if frame is not None:
            yield frame

cams = [
    Camera(cv2.VideoCapture(f"{RTSP_SERVER}/{i}"))
    for i in range(N_CAMS)
]


if __name__ == "__main__":
    import time

    print("Starting frame stream. Press 'q' to quit.")
    for frame in stream_last_frame(cams[0]):
        # Display the frame
        cv2.imshow("Frame", frame)
        
        # Check for key press ('q' to quit)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        time.sleep(1)  # Simulate ~1 FPS
    
    cv2.destroyAllWindows()