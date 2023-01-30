import cv2
import mediapipe as mp
import numpy as np
import argparse
import os

def slope(a,b,c,d):
    # avoid devision by zero
    if a == c: 
        return 0
    return (d - b)/(c - a)
    
def detectBelt(gray, name):

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cv2.imwrite(name + "01_gray.jpg", gray)
    gray = clahe.apply(gray)
    cv2.imwrite(name + "02_clahe.jpg", gray)
    # Bluring The Image For Smoothness
    blur = cv2.blur(gray, (1, 1))
    cv2.imwrite(name + "03_blur.jpg", blur)
    # Converting Image To Edges
    edges = cv2.Canny(blur, 50, 200)

    print(edges)

    cv2.imwrite(name + "04_edges.jpg", edges)


    contours, hierarchy = cv2.findContours(image=edges, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    # draw contours on the original image
    image_copy = edges.copy()
    image_copy = cv2.cvtColor(image_copy, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.imwrite(name + '06_contours.jpg', image_copy)


    lines = cv2.HoughLinesP(edges, 1, np.pi / 270, 30, maxLineGap = 10, minLineLength = 100)
    # possible lines for belt
    candidates = []
    if lines is not None:
        for line in lines:
            # we need to detect two lines
            # Co-ordinates Of Current Line
            x1, y1, x2, y2 = line[0]
            cv2.line(gray, (x1, y1), (x2, y2), 0, 3)
            # Slope Of Current Line
            s = slope(x1, y1, x2, y2)
            print("Slope: " + str(s) + " " + name)
            # fine tune parameters for slope
            if (s < 2 and s > 1):
                print("seat belt found")
                candidates.append(line[0])
    else:
        print("No lines detected")
    cv2.imwrite(name + "05_lines.jpg", gray)
    return candidates

parser = argparse.ArgumentParser(description='Find seatbelt')
parser.add_argument('--input', help='Path to input image.')
args = parser.parse_args()

# initialize Pose estimator
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

frame = cv2.imread(args.input)

name = os.path.splitext(args.input)[0]

# resize image
width, height, dims = frame.shape
scale = 640 / width
width = int(frame.shape[1] * scale)
height = int(frame.shape[0] * scale)
dim = (width, height)
frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

# convert the frame to RGB format
rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# process the RGB frame to get the result
results = pose.process(rgb_image)

image_hight, image_width, _ = frame.shape
lx = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image_width)
ly = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_hight)
rx = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image_width)
ry = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_hight)

# draw a line from shoulder to shoulder
cv2.line(frame, (rx, ry), (lx, ly), (255, 255, 255), 3)

# pass subimage to detect seat belt
dy = max(ry, ly)
# it can be easier to detect seat belt with full image (seat belt can easily be detected from top holder)
# (red_image, green_image, blue_image) = cv2.split(rgb_image)
(red_image, green_image, blue_image) = cv2.split(rgb_image[dy:, rx:lx, :])
for candidate in detectBelt(red_image, name + "_r_"):
    x1, y1, x2, y2 = candidate
    cv2.line(frame, (x1 + rx, y1 + dy), (x2+ rx, y2 + dy), (0, 0, 255), 3)
for candidate in detectBelt(green_image, name + "_g_"):
    x1, y1, x2, y2 = candidate
    cv2.line(frame, (x1 + rx, y1 + dy), (x2+ rx, y2 + dy), (0, 255, 0), 3)
for candidate in detectBelt(blue_image, name + "_b_"):
    x1, y1, x2, y2 = candidate
    cv2.line(frame, (x1 + rx, y1 + dy), (x2+ rx, y2 + dy), (255, 0, 0), 3)
# show the final output
cv2.imwrite(name + "_output.jpg", frame)
