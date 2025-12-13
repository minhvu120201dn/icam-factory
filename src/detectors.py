import cv2
import numpy as np
from typing import Tuple, List, Optional
from ultralytics.engine.results import Results

from alerts import save_alert


class DangerZoneDetector:
    """Detect when people enter a dangerous zone."""
    
    def __init__(self, camera_id: int, zone_polygon: List[Tuple[int, int]]):
        """Initialize the danger zone detector.
        
        Args:
            camera_id: The camera ID for alerts
            zone_polygon: List of (x, y) points defining the danger zone polygon
        """
        self.camera_id = camera_id
        self.zone_polygon = np.array(zone_polygon, dtype=np.int32)
        self.alerted_tracks = set()  # Track IDs that have already triggered alerts
        
    def check_zone(self, results: Results, frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
        """Check if any person is in the danger zone.
        
        Args:
            results: YOLO detection results
            frame: The current frame
            
        Returns:
            Frame with danger zone visualization
        """
        # Draw danger zone
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.zone_polygon], (0, 0, 255))
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        cv2.polylines(frame, [self.zone_polygon], True, (0, 0, 255), 3)
        
        # Check detections
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            
            for i, box in enumerate(boxes):
                # Get class ID and track ID
                cls_id = int(box.cls[0]) if box.cls is not None else None
                track_id = int(box.id[0]) if box.id is not None else None
                
                # Only check for person class (class 0 in COCO)
                if cls_id == 2:  # Assuming class 2 is person in your model
                    # Get bounding box center point
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    # Check if center point is inside danger zone
                    point_in_zone = cv2.pointPolygonTest(
                        self.zone_polygon, 
                        (center_x, center_y), 
                        False
                    ) >= 0
                    
                    if point_in_zone:
                        # Draw warning on frame
                        cv2.circle(frame, (center_x, center_y), 10, (0, 0, 255), -1)
                        cv2.putText(
                            frame, 
                            "DANGER!", 
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 0, 255),
                            2
                        )
                        
                        # Trigger alert if not already alerted for this track
                        if track_id is not None and track_id not in self.alerted_tracks:
                            save_alert(
                                camera_id=self.camera_id,
                                alert_type="danger_zone",
                                track_id=track_id,
                                frame=frame,
                                details=f"Person entered danger zone at ({center_x}, {center_y})"
                            )
                            self.alerted_tracks.add(track_id)
        
        return frame


class NoHelmetDetector:
    """Detect people without helmets."""
    
    def __init__(self, camera_id: int):
        """Initialize the no helmet detector.
        
        Args:
            camera_id: The camera ID for alerts
        """
        self.camera_id = camera_id
        self.alerted_tracks = set()  # Track IDs that have already triggered alerts
        
    def check_helmet(self, results: Results, frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
        """Check if any person is without a helmet.
        
        Args:
            results: YOLO detection results
            frame: The current frame
            
        Returns:
            Frame with helmet detection visualization
        """
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            
            # Track which person IDs have helmets nearby
            person_boxes = []
            helmet_boxes = []
            
            for box in boxes:
                cls_id = int(box.cls[0]) if box.cls is not None else None
                
                # Class 0: helmet, Class 1: head, Class 2: person (adjust based on your model)
                if cls_id == 2:  # person
                    person_boxes.append(box)
                elif cls_id == 0:  # helmet
                    helmet_boxes.append(box)
            
            # Check each person for helmet
            for person_box in person_boxes:
                track_id = int(person_box.id[0]) if person_box.id is not None else None
                x1, y1, x2, y2 = person_box.xyxy[0].cpu().numpy()
                
                # Check if there's a helmet near this person's head area
                has_helmet = False
                for helmet_box in helmet_boxes:
                    hx1, hy1, hx2, hy2 = helmet_box.xyxy[0].cpu().numpy()
                    
                    # Check if helmet box overlaps with upper part of person box
                    if (hx1 < x2 and hx2 > x1 and 
                        hy1 < y1 + (y2 - y1) * 0.3 and hy2 > y1):
                        has_helmet = True
                        break
                
                if not has_helmet:
                    # Draw warning on frame
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                    cv2.putText(
                        frame,
                        "NO HELMET!",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 255),
                        2
                    )
                    
                    # Trigger alert if not already alerted for this track
                    if track_id is not None and track_id not in self.alerted_tracks:
                        save_alert(
                            camera_id=self.camera_id,
                            alert_type="no_helmet",
                            track_id=track_id,
                            frame=frame,
                            details=f"Person without helmet detected"
                        )
                        self.alerted_tracks.add(track_id)
        
        return frame
