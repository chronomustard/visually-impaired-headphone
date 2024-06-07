import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import math
import requests
import threading
import supervision as sv
from ultralytics import YOLO
from playsound import playsound

TELE_TOKEN = "7277909972:AAFjj5meKrGqF5msuvmaxUx5prRxseIPWdE"
CHAT_ID = "5634015200"
message = "Help requested!"
url = f"https://api.telegram.org/bot{TELE_TOKEN}/sendMessage?chat_id={CHAT_ID}&text={message}"
print(requests.get(url).json())

def play_sound_in_thread(sound_file):
    # Function to play sound in a separate thread
    thread = threading.Thread(target=playsound, args=(sound_file,))
    thread.start()

# Define region points using the provided coordinates in numpy array format
ZONE_POLYGON = np.array([
    [160, 280],  # Top left
    [460, 280],  # Top right
    [570, 472],  # Bottom right
    [70, 472]    # Bottom left
])

threshold_counter = 150
model = YOLO('asset/model/best.onnx')
vidcap = cv2.VideoCapture("asset/video/video_1.mp4")
success, image = vidcap.read()

def nothing(x):
    pass

# cv2.namedWindow("Trackbars")
# cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
# cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
# cv2.createTrackbar("L - V", "Trackbars", 200, 255, nothing)
# cv2.createTrackbar("U - H", "Trackbars", 255, 255, nothing)
# cv2.createTrackbar("U - S", "Trackbars", 50, 255, nothing)
# cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

L_counter = 0
R_counter = 0
S_counter = 0
N_counter = 0

text = "Determining Path"

zone = sv.PolygonZone(polygon=ZONE_POLYGON, frame_resolution_wh= (640,480))
zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.RED)

while True:
    
    success, image = vidcap.read()

    # image = cv2.imread("image3.png")

    frame = cv2.resize(image, (640,480))

    result = model(frame, verbose = False)[0]
    detections = sv.Detections.from_ultralytics(result)

    tl = (160,280)
    bl = (70,472)
    tr = (460,280)
    br = (570,472)

    cv2.circle(frame, tl, 5, (0,0,255), -1)
    cv2.circle(frame, bl, 5, (0,0,255), -1)
    cv2.circle(frame, tr, 5, (0,0,255), -1)
    cv2.circle(frame, br, 5, (0,0,255), -1)

    pts1 = np.float32([tl,bl,tr,br])
    pts2 = np.float32([[0,0],[0,480],[640,0],[640,480]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    transformed_frame = cv2.warpPerspective(frame, matrix, (640,480))
    hsv_transformed_frame = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2HSV)

    # USE THIS FOR TUNING THRESHOLD
    # l_h = cv2.getTrackbarPos("L - H", "Trackbars") 
    # l_s = cv2.getTrackbarPos("L - S", "Trackbars") 
    # l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    # u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    # u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    # u_v = cv2.getTrackbarPos("U - V", "Trackbars")

    l_h = 15
    l_s = 50
    l_v = 50
    u_h = 114
    u_s = 141
    u_v = 141

    lower = np.array([l_h,l_s,l_v])
    upper = np.array([u_h,u_s,u_v])
    mask = cv2.inRange(hsv_transformed_frame, lower, upper)

    #Histogram
    histogram = np.sum(mask[mask.shape[0]//2:, :], axis=0)
    midpoint = np.int64(histogram.shape[0]/2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    #Sliding Window
    y = 472
    lx = []
    rx = []

    msk = mask.copy()

    while y>0:
        ## Left threshold
        img = mask[y-40:y, left_base-50:left_base+50]
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                lx.append(left_base-50 + cx)
                left_base = left_base-50 + cx
        
        ## Right threshold
        img = mask[y-40:y, right_base-50:right_base+50]
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                rx.append(right_base-50 + cx)
                right_base = right_base-50 + cx
        
        avg_rx = np.mean(rx)
        print(avg_rx, "|| ", L_counter, S_counter, R_counter, N_counter)
        
        cv2.rectangle(msk, (left_base-50,y), (left_base+50,y-40), (255,255,255), 2)
        cv2.rectangle(msk, (right_base-50,y), (right_base+50,y-40), (255,255,255), 2)
        y -= 40

        if avg_rx == 0 or math.isnan(avg_rx):
            N_counter = N_counter + 1
        else:
            if avg_rx < 180:
                L_counter = L_counter + 1
            elif (avg_rx >= 180) and (avg_rx <= 460):
                S_counter = S_counter + 1
            elif avg_rx > 460:
                R_counter = R_counter + 1

        if L_counter > threshold_counter:
            if text != "Go Left":
                play_sound_in_thread('asset/audio/left.mp3')  # Play sound in parallel
            text = "Go Left"
            L_counter = 0
            R_counter = 0
            S_counter = 0
            N_counter = 0
        
        if R_counter > threshold_counter:
            if text != "Go Right":
                play_sound_in_thread('asset/audio/right.mp3')  # Play sound in parallel
            text = "Go Right"
            L_counter = 0
            R_counter = 0
            S_counter = 0
            N_counter = 0

        if S_counter > threshold_counter:
            if text != "Go Straight":
                play_sound_in_thread('asset/audio/straight.mp3')  # Play sound in parallel
            text = "Go Straight"
            L_counter = 0
            R_counter = 0
            S_counter = 0
            N_counter = 0

        if N_counter > round(threshold_counter*2):
            rx = []
            if text != "No Path Found":
                play_sound_in_thread('asset/audio/nopath.mp3')  # Play sound in parallel
            text = "No Path Found"
            L_counter = 0
            R_counter = 0
            S_counter = 0
            N_counter = 0

        font                   = cv2.FONT_HERSHEY_SIMPLEX
        topLeftCornerOfText     = (10,35)
        fontScale              = 0.8
        fontColor              = (255,255,255)
        thickness              = 1
        lineType               = 2

        cv2.putText(msk,text, 
            topLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)
        
        cv2.putText(frame,text, 
            topLeftCornerOfText, 
            font, 
            fontScale,
            (20,255,20),
            2,
            3)
        
        zone.trigger(detections=detections)
        annotated_frame = zone_annotator.annotate(scene=frame.copy())


    cv2.imshow("Original", annotated_frame)
    cv2.imshow("Transformed", transformed_frame)
    cv2.imshow("Path Detection", mask)
    cv2.imshow("Path Detection - Sliding Windows", msk)

    if cv2.waitKey(10) == 27:
        break