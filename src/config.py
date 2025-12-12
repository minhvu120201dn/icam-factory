import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

RTSP_SERVER = os.getenv("RTSP_SERVER", "rtsp://localhost:8554")
N_CAMS = int(os.getenv("N_CAMS", 3))