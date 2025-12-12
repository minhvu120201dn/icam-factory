# Variables
INPUT_VIDEO ?= datasets/OxfordTownCentre/TownCentreXVID.mp4
RTSP_URL ?= rtsp://localhost:8554/0

rtsp_server:
	ffmpeg \
	-stream_loop -1 -re \
	-i $(INPUT_VIDEO) \
	-c copy \
	-rtsp_transport tcp \
	-f rtsp $(RTSP_URL)

rtsp_client:
	ffplay $(RTSP_URL)
