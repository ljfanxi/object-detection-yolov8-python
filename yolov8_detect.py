import datetime
from ultralytics import YOLO
import cv2

CONFIDENCE_THRESHOLD = 0.8
GREEN = (0, 255, 0)
WHITE = (0, 0, 255)

# initialize the video capture object
video_cap = cv2.VideoCapture("video.mp4")
new_width = 640  # Set your desired width
new_height = 900  # Set your desired height


# load the pre-trained YOLOv8n model
model = YOLO("best.pt")

while True:
    start = datetime.datetime.now()

    ret, frame = video_cap.read()

    if not ret:
        break

    # run the YOLO model on the frame
    detections = model(frame)[0]

    # initialize the list of bounding boxes and confidences
    results = []

    # loop over the detections
    for data in detections.boxes.data.tolist():
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = data[4]

        # filter out weak detections by ensuring the
        # confidence is greater than the minimum confidence
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue

        # if the confidence is greater than the minimum confidence,
        # get the bounding box and the class id
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        class_id = int(data[5])
        # add the bounding box (x, y, w, h), confidence, and class id to the results list
        results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])

    # loop over the results and draw bounding boxes on the frame
    for result in results:
        bbox, confidence, class_id = result
        xmin, ymin, width, height = map(int, bbox)
        cv2.rectangle(frame, (xmin, ymin), (xmin + width, ymin + height), GREEN, 2)
        class_name = 'king'     
        cv2.putText(frame, class_name, (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

    # end time to compute the fps
    end = datetime.datetime.now()
    # show the time it took to process 1 frame
    print(f"Time to process 1 frame: {(end - start).total_seconds() * 1000:.0f} milliseconds")
    # calculate the frame per second and draw it on the frame
    fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
    cv2.putText(frame, fps, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

    resized_frame = cv2.resize(frame, (new_width, new_height))

    # show the frame to our screen
    cv2.imshow("Frame", resized_frame)

    if cv2.waitKey(1) == ord("q"):
        break

video_cap.release()
cv2.destroyAllWindows()
