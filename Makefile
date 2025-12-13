# Variables
INPUT_VIDEO_0 ?= datasets/TownCentre/input_video_s2_l1_01.mp4
INPUT_VIDEO_1 ?= datasets/TownCentre/input_video_s2_l1_03.mp4
INPUT_VIDEO_2 ?= datasets/TownCentre/input_video_s2_l1_04.mp4
RTSP_BASE_URL ?= rtsp://localhost:8554

# FFmpeg command options
FFMPEG_OPTS = -stream_loop -1 -re
FFMPEG_OUTPUT_OPTS = -c copy -rtsp_transport tcp -f rtsp

# Start all 3 RTSP servers in parallel
rtsp_servers:
	@echo "Starting 3 RTSP servers..."
	@ffmpeg $(FFMPEG_OPTS) -i $(INPUT_VIDEO_0) $(FFMPEG_OUTPUT_OPTS) $(RTSP_BASE_URL)/0 & \
	ffmpeg $(FFMPEG_OPTS) -i $(INPUT_VIDEO_1) $(FFMPEG_OUTPUT_OPTS) $(RTSP_BASE_URL)/1 & \
	ffmpeg $(FFMPEG_OPTS) -i $(INPUT_VIDEO_2) $(FFMPEG_OUTPUT_OPTS) $(RTSP_BASE_URL)/2 & \
	wait

# Start individual RTSP servers
rtsp_server_0:
	ffmpeg $(FFMPEG_OPTS) -i $(INPUT_VIDEO_0) $(FFMPEG_OUTPUT_OPTS) $(RTSP_BASE_URL)/0

rtsp_server_1:
	ffmpeg $(FFMPEG_OPTS) -i $(INPUT_VIDEO_1) $(FFMPEG_OUTPUT_OPTS) $(RTSP_BASE_URL)/1

rtsp_server_2:
	ffmpeg $(FFMPEG_OPTS) -i $(INPUT_VIDEO_2) $(FFMPEG_OUTPUT_OPTS) $(RTSP_BASE_URL)/2

# View RTSP clients
rtsp_client_0:
	ffplay $(RTSP_BASE_URL)/0

rtsp_client_1:
	ffplay $(RTSP_BASE_URL)/1

rtsp_client_2:
	ffplay $(RTSP_BASE_URL)/2
