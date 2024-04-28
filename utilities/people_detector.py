#  main.py

import cv2
import numpy as np
import imutils
import os
import datetime

# Get the path to the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Paths to the prototxt and caffemodel files
protopth = os.path.join(script_dir, "MobileNetSSD_deploy.prototxt")
modelpth = os.path.join(script_dir, "MobileNetSSD_deploy.caffemodel")

detector = cv2.dnn.readNetFromCaffe(prototxt=protopth, caffeModel=modelpth)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
           "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
           "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def non_max_suppression(boxes, overlap_thresh=0.3):
    if len(boxes) == 0:
        return []

    # Initialize the list of picked indices
    pick = []

    # Grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Compute the area of the bounding boxes
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    # Sort the bounding boxes by their bottom-right y-coordinate
    idxs = np.argsort(y2)

    # Keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # Grab the last index in the indexes list and add the index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # Find the largest (x, y) coordinates for the start of the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # Compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # Delete all indexes from the index list that have overlap greater than the specified overlap threshold
        idxs = np.delete(idxs, np.concatenate(
            ([last], np.where(overlap > overlap_thresh)[0])))

    # Return only the bounding boxes that were picked
    return boxes[pick]


def detect_persons_in_video(video_path, counter):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps_start_time = datetime.datetime.now()
    total_frames = 0
    frame_skip = 3  # Process every 5th frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1

        if total_frames % frame_skip != 0:
            continue  # Skip this frame

        frame = imutils.resize(frame, width=640)

        (H, W) = frame.shape[:2]

        blb = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        detector.setInput(blb)
        per_detect = detector.forward()

        boxes = []  # Initialize list to store bounding box coordinates

        count = 0  # Initialize count of persons detected in the current frame

        for i in np.arange(0, per_detect.shape[2]):
            confidence = per_detect[0, 0, i, 2]
            if confidence > 0.1:
                index = int(per_detect[0, 0, i, 1])

                if CLASSES[index] == "person":
                    count += 1  # Increment count when a person is detected

                    p_box = per_detect[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = p_box.astype('int')

                    # Append bounding box coordinates
                    boxes.append([startX, startY, endX, endY])

        # Apply Non-Maximum Suppression
        boxes = np.array(boxes)
        nms_boxes = non_max_suppression(boxes)

        for (startX, startY, endX, endY) in nms_boxes:
            cv2.rectangle(frame, (startX, startY),
                          (endX, endY), (0, 0, 255), 2)

        # Increment the global counter with the count of persons detected in the current frame
        counter.increment(count)

        # Display count of persons detected in the current frame
        cv2.putText(frame, f"Persons in frame: {count}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Display overall count of persons detected
        # cv2.putText(frame, f"Total persons detected: {counter.get_count()}",
        #            (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (155, 200, 0), 2)

        # Calculate average FPS and display it on the frame
        fps_end_time = datetime.datetime.now()
        time_diff = fps_end_time - fps_start_time
        if time_diff.seconds == 0:
            fps = 0.0
        else:
            fps = (total_frames / time_diff.seconds)
        fps_text = f"Average FPS: {fps:.2f}"
        cv2.putText(frame, fps_text, (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Display the modified frame
        cv2.imshow("Result", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def detect_persons_in_image(image_path, counter):
    frame = cv2.imread(image_path)

    frame = imutils.resize(frame, width=525)
    (H, W) = frame.shape[:2]

    blb = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
    detector.setInput(blb)
    per_detect = detector.forward()

    # Initialize the count of detected persons
    count = 0

    for i in np.arange(0, per_detect.shape[2]):
        confidence = per_detect[0, 0, i, 2]
        if confidence > 0.5:
            index = int(per_detect[0, 0, i, 1])

            if CLASSES[index] == "person":
                count += 1  # Increment count when a person is detected

                # Extract coordinates for the bounding box
                startX, startY, endX, endY = (
                    per_detect[0, 0, i, 3:7] * np.array([W, H, W, H])).astype("int")

                # Calculate center coordinates of the bounding box
                centerX = int((startX + endX) / 2)
                centerY = int((startY + endY) / 2)

                # Draw the bounding box rectangle
                cv2.rectangle(frame, (startX, startY),
                              (endX, endY), (0, 255, 0), 2)

                # Draw a red dot at the center of the bounding box
                cv2.circle(frame, (centerX, centerY), 5, (0, 0, 255), -1)

    # Increment the counter with the count of detected persons
    counter.increment(count)

    # Put text on the image indicating the count of detected persons
    cv2.putText(frame, f"Persons detected: {
                count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 255, 100), 2)

    # Display the modified image
    cv2.imshow("Modified Image", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return count
