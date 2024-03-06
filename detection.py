import cv2
from main import detect_and_draw
from norfair import Detection, Tracker, Video, draw_tracked_objects




def save_video_with_bounding_boxes(input_video_path, output_video_path):
    # Open the input video
    video_capture = cv2.VideoCapture(input_video_path)
    # norfair
    video = Video(input_path=input_video_path)

    # Get the video's frame width, height, and frames per second
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))

    # Create VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while True:
        # Read the next frame
        ret, frame = video_capture.read()

        if not ret:
            break

        # Perform object detection on the frame
        # Replace the following line with your own object detection code
        # The model variable should contain the logic to detect objects and draw bounding boxes
        # print(frame)
        # frame = tf.convert_to_tensor(frame, dtype=tf.int8)
        # print("converted_frame")
        (im_height, im_width, _)=frame.shape
        # print(type(frame))
        # frame = frame.reshape(1,im_height,im_width, 3).astype(np.uint8)
        # print(frame.shape)
        # return
        # detected_frame =run_detector(model, frame,im_height,im_width)
        detected_frame =detect_and_draw(frame)

        output_video.write(detected_frame)
        # Display the frame with bounding boxes
        # cv2.imshow('Video with Bounding Boxes', detected_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and output video objects
    video_capture.release()
    output_video.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()



# Example usage
input_video_path = 'videos/cars_1.mp4'
output_video_path = 'videos/cars_1_Tout.mp4'

save_video_with_bounding_boxes(input_video_path, output_video_path)