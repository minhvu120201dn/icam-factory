import sqlite3
from datetime import datetime
from typing import List, Literal
import cv2
from pathlib import Path

# Initialize database
DB_PATH = Path(__file__).parent.parent / "alerts/alerts.db"
SNAPSHOTS_DIR = Path(__file__).parent.parent / "alerts/snapshots"
SNAPSHOTS_DIR.mkdir(exist_ok=True)

def init_database():
    """Initialize the alerts database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id INTEGER NOT NULL,
            alert_type TEXT NOT NULL,
            track_id INTEGER,
            timestamp TEXT NOT NULL,
            snapshot_path TEXT,
            details TEXT
        )
    """)
    
    conn.commit()
    conn.close()

def save_alert(
    camera_id: int,
    alert_type: Literal["danger_zone", "no_helmet"],
    track_id: int = None,
    frame: cv2.typing.MatLike = None,
    details: str = None
):
    """Save an alert to the database and trigger notification.
    
    Args:
        camera_id: The camera that detected the alert
        alert_type: Type of alert ("danger_zone" or "no_helmet")
        track_id: The tracking ID of the object
        frame: The frame to save as snapshot (optional)
        details: Additional details about the alert
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    timestamp = datetime.now().isoformat()
    snapshot_path = None
    
    # Save snapshot if frame provided
    if frame is not None:
        snapshot_filename = f"cam{camera_id}_{alert_type}_track{track_id}_{timestamp.replace(':', '-')}.jpg"
        snapshot_path = str(SNAPSHOTS_DIR / snapshot_filename)
        cv2.imwrite(snapshot_path, frame)
    
    cursor.execute("""
        INSERT INTO alerts (camera_id, alert_type, track_id, timestamp, snapshot_path, details)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (camera_id, alert_type, track_id, timestamp, snapshot_path, details))
    
    conn.commit()
    conn.close()
    
    # Trigger alert notification
    trigger_alert(camera_id, alert_type, track_id, timestamp)

def trigger_alert(camera_id: int, alert_type: str, track_id: int, timestamp: str):
    """Trigger an alert notification.
    
    Args:
        camera_id: The camera that detected the alert
        alert_type: Type of alert
        track_id: The tracking ID of the object
        timestamp: When the alert occurred
    """
    alert_messages = {
        "danger_zone": f"⚠️  DANGER ZONE BREACH - Camera {camera_id}, Track ID {track_id}",
        "no_helmet": f"⚠️  NO HELMET DETECTED - Camera {camera_id}, Track ID {track_id}"
    }
    
    message = alert_messages.get(alert_type, f"Alert from Camera {camera_id}")
    print(f"\n{'='*60}")
    print(f"🚨 ALERT: {message}")
    print(f"Time: {timestamp}")
    print(f"{'='*60}\n")
    
    # Here you can add additional notification methods:
    # - Send email
    # - Send SMS
    # - Push notification
    # - Webhook to monitoring system
    # - etc.

def get_recent_alerts(camera_id: int = None, limit: int = 10) -> List[dict]:
    """Get recent alerts from the database.
    
    Args:
        camera_id: Filter by camera ID (optional)
        limit: Maximum number of alerts to return
        
    Returns:
        List of alert dictionaries
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if camera_id is not None:
        cursor.execute("""
            SELECT id, camera_id, alert_type, track_id, timestamp, snapshot_path, details
            FROM alerts
            WHERE camera_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (camera_id, limit))
    else:
        cursor.execute("""
            SELECT id, camera_id, alert_type, track_id, timestamp, snapshot_path, details
            FROM alerts
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    alerts = []
    for row in rows:
        alerts.append({
            "id": row[0],
            "camera_id": row[1],
            "alert_type": row[2],
            "track_id": row[3],
            "timestamp": row[4],
            "snapshot_path": row[5],
            "details": row[6]
        })
    
    return alerts

# Initialize database on module import
init_database()
