ffmpeg -f lavfi -i testsrc=duration=10:size=1280x720:rate=30 -c:v libx264 -profile:v high -level 4.0 -pix_fmt yuv420p data/test.mp4
echo "Not a video" > data/not_a_video.txt
