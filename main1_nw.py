import cv2
import numpy as np
from scipy.spatial import distance as dist
from imutils.video import VideoStream
import imutils
import time
from pygame import mixer
import matplotlib.pyplot as plt
from EAR import eye_aspect_ratio
from MAR import mouth_aspect_ratio
from HeadPose import getHeadTiltAndCoords

mixer.init()
mixer.music.load("music.wav")

print("[INFO] loading models...")
# Load the pre-trained SSD model for face detection
net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")

# Load the pre-trained facial landmarks model
landmark_net = cv2.dnn.readNetFromCaffe("VGG_FACE_deploy.prototxt", "VGG_FACE.caffemodel")

print("[INFO] initializing camera...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

frame_width = 1024
frame_height = 576

image_points = np.array([
    (359, 391),
    (399, 561),
    (337, 297),
    (513, 301),
    (345, 465),
    (453, 469)
], dtype="double")

EYE_AR_THRESH = 0.25
MOUTH_AR_THRESH = 0.79
EYE_AR_CONSEC_FRAMES = 7
COUNTER = 0
(mStart, mEnd) = (49, 68)

def get_landmarks(frame, box):
    (startX, startY, endX, endY) = box
    face = frame[startY:endY, startX:endX]
    face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (48, 48), (0, 0, 0), swapRB=True, crop=False)
    landmark_net.setInput(face_blob)
    landmarks = landmark_net.forward()

    h, w = face.shape[:2]
    landmarks = landmarks.reshape(-1, 2)
    landmarks = np.column_stack((landmarks[:, 0] * w, landmarks[:, 1] * h))
    landmarks[:, 0] += startX
    landmarks[:, 1] += startY
    return landmarks

while True:
    frame = vs.read()
    
    frame = imutils.resize(frame, width=1024, height=576)
    frame1 = frame.copy()
    frame1 = imutils.resize(frame1, width=450, height=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    colormap = plt.get_cmap('inferno')
    heatmap = (colormap(gray) * 2**16).astype(np.uint16)[:, :, :3]
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

    size = gray.shape

    # Prepare input blob and perform forward pass to get face detections
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    if detections.shape[2] > 0:
        # Extract the confidence (i.e., probability) associated with the detection
        confidence = detections[0, 0, 0, 2]

        # Filter out weak detections
        if confidence > 0.5:
            box = detections[0, 0, 0, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")

            # Get landmarks
            landmarks = get_landmarks(frame, (startX, startY, endX, endY))
            
            # Display the number of faces found
            text = "1 face found"
            cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Create a mask for the background
            mask = np.zeros_like(frame1)

            (bX, bY, bW, bH) = (startX, startY, endX - startX, endY - startY)

            # Draw circle for face detection in frame
            cv2.circle(frame, (bX + int(bW / 2), bY + int(bH / 2)), 80, (255, 0, 255), 2)

            # Create a mask for the face region
            mask[bY:bY + bH, bX:bX + bW] = frame1[bY:bY + bH, bX:bX + bW]

            # Apply the mask to the frame to remove the background
            frame1 = cv2.bitwise_and(frame1, mask)

            # Create circular frame on the right side of the main frame
            small_frame_height = 250
            small_frame_width = 250
            radius = small_frame_width // 2
            center = (frame.shape[1] - radius, radius)

            face_region = frame[bY:bY + bH, bX:bX + bW]
            face_region_resized = cv2.resize(face_region, (small_frame_width, small_frame_height))

            leftEye = landmarks[36:42]
            rightEye = landmarks[42:48]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            if ear < EYE_AR_THRESH:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    cv2.putText(frame, "Eyes Closed!", (500, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    print("Eyes closed")
                    mixer.music.play()
            else:
                COUNTER = 0

            mouth = landmarks[mStart:mEnd]
            mouthMAR = mouth_aspect_ratio(mouth)
            mar = mouthMAR

            if mar > MOUTH_AR_THRESH:
                cv2.putText(frame, "Yawning!", (800, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                mixer.music.play()

            for (i, (x, y)) in enumerate(landmarks):
                if i == 33:
                    image_points[0] = np.array([x, y], dtype='double')
                elif i == 8:
                    image_points[1] = np.array([x, y], dtype='double')
                elif i == 36:
                    image_points[2] = np.array([x, y], dtype='double')
                elif i == 45:
                    image_points[3] = np.array([x, y], dtype='double')
                elif i == 48:
                    image_points[4] = np.array([x, y], dtype='double')
                elif i == 54:
                    image_points[5] = np.array([x, y], dtype='double')

            (head_tilt_degree, start_point, end_point, end_point_alt) = getHeadTiltAndCoords(size, image_points, frame_height)
            if head_tilt_degree:
                cv2.putText(frame, 'Head Tilt Degree: ' + str(head_tilt_degree[0]), (170, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "No Face!", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        print("No Face Found")
        #mixer.music.play()

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
