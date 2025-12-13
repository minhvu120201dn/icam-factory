"""
Interactive tool to define danger zones by clicking on video frames.
"""
import cv2
import json
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np

from stream import CameraStreamer
from config import RTSP_SERVER, N_CAMS


class ZoneDefiner:
    """Interactive tool to define danger zones by clicking on frames."""
    
    def __init__(self, camera_id: int, streamer: CameraStreamer):
        """Initialize the zone definer.
        
        Args:
            camera_id: The camera ID
            streamer: CameraStreamer instance
        """
        self.camera_id = camera_id
        self.streamer = streamer
        self.window_name = f"Define Zone - Camera {camera_id}"
        self.points: List[Tuple[int, int]] = []
        self.frame = None
        self.original_frame = None
        self.is_complete = False
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for point selection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add point on left click
            self.points.append((x, y))
            print(f"Point {len(self.points)}: ({x}, {y})")
            self.draw_zone()
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Remove last point on right click
            if self.points:
                removed = self.points.pop()
                print(f"Removed point: {removed}")
                self.draw_zone()
    
    def draw_zone(self):
        """Draw the current zone on the frame."""
        if self.original_frame is None:
            return
        
        # Start with original frame
        self.frame = self.original_frame.copy()
        
        if len(self.points) == 0:
            return
        
        # Draw points
        for i, point in enumerate(self.points):
            cv2.circle(self.frame, point, 5, (0, 255, 0), -1)
            cv2.putText(
                self.frame,
                str(i + 1),
                (point[0] + 10, point[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
        
        # Draw lines between points
        if len(self.points) > 1:
            for i in range(len(self.points) - 1):
                cv2.line(self.frame, self.points[i], self.points[i + 1], (0, 255, 0), 2)
        
        # Draw polygon if we have at least 3 points
        if len(self.points) >= 3:
            # Close the polygon
            cv2.line(self.frame, self.points[-1], self.points[0], (0, 255, 0), 2)
            
            # Fill polygon with transparency
            overlay = self.frame.copy()
            pts = np.array(self.points, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (0, 0, 255))
            self.frame = cv2.addWeighted(self.frame, 0.7, overlay, 0.3, 0)
        
        # Display instructions
        instructions = [
            "Left Click: Add point",
            "Right Click: Remove last point",
            "SPACE: Get new frame",
            "ENTER: Finish (min 3 points)",
            "ESC: Cancel",
            f"Points: {len(self.points)}"
        ]
        
        y_offset = 30
        for instruction in instructions:
            cv2.putText(
                self.frame,
                instruction,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            cv2.putText(
                self.frame,
                instruction,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1
            )
            y_offset += 25
    
    def define_zone(self) -> List[Tuple[int, int]]:
        """Start the interactive zone definition process.
        
        Returns:
            List of (x, y) points defining the zone polygon, or empty list if cancelled
        """
        print(f"\n{'='*60}")
        print(f"Defining danger zone for Camera {self.camera_id}")
        print(f"{'='*60}")
        print("Instructions:")
        print("  - Left click to add points")
        print("  - Right click to remove last point")
        print("  - Press SPACE to get a new frame")
        print("  - Press ENTER to finish (minimum 3 points)")
        print("  - Press ESC to cancel")
        print(f"{'='*60}\n")
        
        # Get a frame from the stream
        stream_gen = self.streamer()
        for frame in stream_gen:
            if frame is not None:
                self.original_frame = frame.copy()
                self.frame = frame.copy()
                break
        
        if self.original_frame is None:
            print("Error: Could not get frame from camera")
            return []
        
        # Create window and set mouse callback
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Main loop
        while True:
            if self.frame is not None:
                cv2.imshow(self.window_name, self.frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # SPACE key - get new frame
            if key == 32:  # SPACE
                print("Getting new frame...")
                # Get next non-blank frame from stream
                for new_frame in stream_gen:
                    if new_frame is not None and new_frame.mean() > 10:  # Check if not blank
                        self.original_frame = new_frame.copy()
                        # Reset points when changing frame
                        self.points = []
                        self.draw_zone()
                        print("✓ New frame loaded (points cleared)")
                        break
            
            # ENTER key - finish
            elif key == 13:  # ENTER
                if len(self.points) >= 3:
                    self.is_complete = True
                    print(f"\n✓ Zone defined with {len(self.points)} points")
                    break
                else:
                    print("Need at least 3 points to define a zone!")
            
            # ESC key - cancel
            elif key == 27:  # ESC
                print("\n✗ Zone definition cancelled")
                self.points = []
                break
        
        cv2.destroyWindow(self.window_name)
        return self.points


def define_all_zones() -> Dict[int, List[Tuple[int, int]]]:
    """Define danger zones for all cameras interactively.
    
    Returns:
        Dictionary mapping camera IDs to zone polygons
    """
    zones = {}
    
    print("\n" + "="*60)
    print("DANGER ZONE DEFINITION TOOL")
    print("="*60)
    print(f"You will define zones for {N_CAMS} cameras")
    print("="*60 + "\n")
    
    for camera_id in range(N_CAMS):
        response = input(f"Define danger zone for Camera {camera_id}? (y/n): ").strip().lower()
        
        if response == 'y':
            streamer = CameraStreamer(f"{RTSP_SERVER}/{camera_id}")
            definer = ZoneDefiner(camera_id, streamer)
            zone_points = definer.define_zone()
            
            if zone_points:
                zones[camera_id] = zone_points
                print(f"Camera {camera_id} zone: {zone_points}\n")
            
            # Clean up streamer
            del streamer
        else:
            print(f"Skipping Camera {camera_id}\n")
    
    return zones


def save_zones_to_file(zones: Dict[int, List[Tuple[int, int]]], filename: str = "danger_zones.json"):
    """Save defined zones to a JSON file.
    
    Args:
        zones: Dictionary of camera IDs to zone polygons
        filename: Output filename
    """
    filepath = Path(__file__).parent.parent / filename
    
    with open(filepath, 'w') as f:
        json.dump(zones, f, indent=2)
    
    print(f"\n✓ Zones saved to: {filepath}")


def load_zones_from_file(filename: str = "danger_zones.json") -> Dict[int, List[Tuple[int, int]]]:
    """Load zones from a JSON file.
    
    Args:
        filename: Input filename
        
    Returns:
        Dictionary of camera IDs to zone polygons
    """
    filepath = Path(__file__).parent.parent / filename
    
    if not filepath.exists():
        return {}
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Convert string keys to integers and tuples
    zones = {}
    for cam_id_str, points in data.items():
        cam_id = int(cam_id_str)
        zones[cam_id] = [tuple(point) for point in points]
    
    return zones


def generate_python_code(zones: Dict[int, List[Tuple[int, int]]]) -> str:
    """Generate Python code for the DANGER_ZONES dictionary.
    
    Args:
        zones: Dictionary of camera IDs to zone polygons
        
    Returns:
        Python code as string
    """
    lines = ["DANGER_ZONES = {"]
    for cam_id, points in zones.items():
        points_str = ", ".join([f"({x}, {y})" for x, y in points])
        lines.append(f"    {cam_id}: [{points_str}],  # Camera {cam_id}")
    lines.append("}")
    
    return "\n".join(lines)


def main():
    """Main function to define zones and save them."""
    print("Starting zone definition tool...")
    
    # Check if zones file exists
    zones_file = Path(__file__).parent.parent / "danger_zones.json"
    if zones_file.exists():
        response = input(f"Zones file exists. Load existing zones? (y/n): ").strip().lower()
        if response == 'y':
            zones = load_zones_from_file()
            print(f"\nLoaded zones for {len(zones)} camera(s):")
            for cam_id, points in zones.items():
                print(f"  Camera {cam_id}: {len(points)} points")
            print()
    
    # Define zones interactively
    zones = define_all_zones()
    
    if zones:
        # Save to JSON file
        save_zones_to_file(zones)
        
        # Generate Python code
        print("\n" + "="*60)
        print("Copy this code to your main.py:")
        print("="*60)
        print(generate_python_code(zones))
        print("="*60 + "\n")
    else:
        print("\nNo zones defined.")


if __name__ == "__main__":
    main()
