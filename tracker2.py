import norfair
import numpy as np
from ultralytics import YOLO
import cv2
from norfair import Detection, Tracker, Video, draw_tracked_objects
import time

total_time = 0
total_frames = 0
detect_time = 0
track_time = 0

tracker = Tracker(distance_function="euclidean", distance_threshold=100)

model = YOLO('yolov8l-world.pt')


def detect(img, model=model):
    # print(img.shape)
    results = model.predict(img)
    return results[0]

def draw(results):
    all_cls = results.names
    boxes = results.boxes
    img = results.orig_img
    image = img.copy()  # Create a copy of the original image
    norfair_detections = []
    for box in boxes:
        cls = all_cls[int(box.cls)]
        conf = box.conf
        if conf.item() < 0.5:
            continue
        xyxy = box.xyxy[0]
        x1,y1,x2,y2 = map(int,xyxy[0:4])
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 4)
        image = cv2.putText(image, cls + "" + str(round(conf.item(), 2)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Calculate the center of the bounding box
        # center = ((x1 + x2) // 2, (y1 + y2) // 2)

        # Create a Detection object with the center of the bounding box
        # detect = Detection(np.array(center))
        # norfair_detections.append(detect)
    #
    # tracked_objects = tracker.update(detections=norfair_detections)
    #
    # # Use draw_points instead of draw_tracked_objects
    norfair.draw_points(image, tracked_objects)

    return image

def detect_and_draw(img):
    global detect_time, track_time
    detect_start = time.time()
    result = detect(img)
    detect_end = time.time()
    detect_time += detect_end - detect_start

    draw_start = time.time()
    img_out = draw(result)
    draw_end = time.time()
    track_time += draw_end - draw_start
    return img_out


def my_detector(frame, model=model):
    results = model(frame)
    return results[0].boxes.xyxy.cpu().detach().numpy()

# Replace 'path/to/video.mp4' with the actual path to your video file
cap = cv2.VideoCapture('vid2.mp4')

# Get the video's width, height, and frames per second (fps)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
out = cv2.VideoWriter('vid2_output.mp4', fourcc, fps, (width, height))

# Function to check if video stream is open correctly
def check_video_stream(cap):
    if not cap.isOpened():
        print("Error opening video stream or file")
        return False
    return True

# Check if video stream is opened successfully
if not check_video_stream(cap):
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if frame is read correctly
    if not ret:
        print("Can't receive frame (stream end?). Exiting...")
        break

    start_time = time.time()
    # Perform detection and drawing on the frame
    frame_out = detect_and_draw(frame)

    end_time = time.time()
    total_time += end_time - start_time
    total_frames += 1
    # Write the processed frame to the output video file
    out.write(frame_out)

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and the VideoWriter
cap.release()
out.release()

print(f"Average time per frame: {total_time / total_frames} seconds")
print(f"Average detection time per frame: {detect_time / total_frames} seconds")
print(f"Average tracking time per frame: {track_time / total_frames} seconds")
print(f"Total time spent: {total_time} seconds")
print(f"Total frames: {total_frames}")