
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import tempfile
from six.moves.urllib.request import urlopen
from six import BytesIO

import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps
import cv2
import time
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


def display_image(image):
  fig = plt.figure(figsize=(20, 15))
  plt.grid(False)
  plt.imshow(image)


def download_and_resize_image(url, new_width=256, new_height=256,
                              display=False):
  _, filename = tempfile.mkstemp(suffix=".jpg")
  response = urlopen(url)
  image_data = response.read()
  image_data = BytesIO(image_data)
  pil_image = Image.open(image_data)
  pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.LANCZOS)
  pil_image_rgb = pil_image.convert("RGB")
  pil_image_rgb.save(filename, format="JPEG", quality=90)
  print("Image downloaded to %s." % filename)
  if display:
    display_image(pil_image)
  return filename


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color,
                               font,
                               thickness=4,
                               display_str_list=()):
  """Adds a bounding box to an image."""
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height)
  draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
             (left, top)],
            width=thickness,
            fill=color)

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_heights = [font.getbbox(ds)[3] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = top + total_display_str_height
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    bbox = font.getbbox(display_str)
    text_width, text_height = bbox[2], bbox[3]
    margin = np.ceil(0.05 * text_height)
    draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                    (left + text_width, text_bottom)],
                   fill=color)
    draw.text((left + margin, text_bottom - text_height - margin),
              display_str,
              fill="black",
              font=font)
    text_bottom -= text_height - 2 * margin


def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
  """Overlay labeled boxes on an image with formatted scores and label names."""
  colors = list(ImageColor.colormap.values())
#   print(class_names)
  try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
                              25)
  except IOError:
    print("Font not found, using default font.")
    font = ImageFont.load_default()

  for i in range(min(boxes.shape[0], max_boxes)):
    if scores[i] >= min_score:
      ymin, xmin, ymax, xmax = tuple(boxes[i])
      display_str = "{}: {}%".format(class_names[i].decode("ascii"),
                                     int(100 * scores[i]))
      color = colors[hash(class_names[i]) % len(colors)]
      image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
      draw_bounding_box_on_image(
          image_pil,
          ymin,
          xmin,
          ymax,
          xmax,
          color,
          font,
          display_str_list=[display_str])
      np.copyto(image, np.array(image_pil))
  return image



module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1" #@param ["https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1", "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"]

detector = hub.load(module_handle).signatures['default']

def load_img(img):
  print("path: ",img)
  img = tf.io.read_file(img)
  print("tf.io.read_file: ",img)
  img = tf.image.decode_jpeg(img, channels=3)
  print("tf.image.decode_jpeg: ", img)
  return img



def run_detector(detector, img):
#   img = load_img(img)
#   print("image: ",img)
  converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
  start_time = time.time()
  result = detector(converted_img)
  end_time = time.time()-start_time
  print("Time per frame: ",end_time)
  result = {key:value.numpy() for key,value in result.items()}

  print("Found %d objects." % len(result["detection_scores"]))
  print("Inference time: ", end_time-start_time)
  image_with_boxes = draw_boxes(
      img, result["detection_boxes"],
      result["detection_class_entities"], result["detection_scores"],min_score=0.05)
  return image_with_boxes



# def run_detector(model,frame,im_height,im_width):

#     img = frame.reshape(1,im_height,im_width, 3).astype(np.uint8)
#     st_time = time.time()
#     result = model(img)
#     print("frame Time: ",time.time() - st_time)
#     out_img = draw_bboxes(img, result)
#     return out_img

def save_video_with_bounding_boxes(input_video_path, output_video_path, model):
    # Open the input video
    video_capture = cv2.VideoCapture(input_video_path)

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
        detected_frame =run_detector(model, frame)

        # print(detected_frame.shape)
        # image1copy = np.uint8(detected_frame)
        # image1=np.reshape(detected_frame,(im_height,im_width,3))
        # print(image1copy.shape)

        # Write the frame with bounding boxes to the output video
        # output_video.write(detected_frame1)
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
input_video_path = '/content/4K_Road_traffic.mp4'
output_video_path = '/content/4K_Road_traffic_detected_5p.mp4'
model = detector  # Replace with your own object detection model

save_video_with_bounding_boxes(input_video_path, output_video_path, model)