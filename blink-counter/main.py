import cv2
import numpy as np
import mediapipe as mp
import math

# Find distance between two points


def findDistance(p1, p2):

    x1, y1 = p1
    x2, y2 = p2
    distance = math.hypot(x2 - x1, y2 - y1)

    return distance


# Detect the Iris of the Eyes and add Sharingan
def detectIrisAndCount(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_h, img_w = image.shape[:2]
    results = FaceMesh.process(rgb_image)

    if not results.multi_face_landmarks:
        return image

    mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                            for p in results.multi_face_landmarks[0].landmark])

    (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
    # (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])

    leftTop = mesh_points[LEFT_EYE_TOP]
    leftBottom = mesh_points[LEFT_EYE_BOTTOM]

    rightTop = mesh_points[RIGHT_EYE_TOP]
    rightBottom = mesh_points[RIGHT_EYE_BOTTOM]

    leftDis = findDistance(leftTop, leftBottom)
    rightDis = findDistance(rightTop, rightBottom)

    # radius * 1.5 = 75% height of iris * irisVisible = 70% of 75% iris is the minimum height
    # we are taking left eye's iris as initial height. why? test results says...
    minEyeDis = (l_radius * 1.5) * irisVisible
    # minEyeDisRight = (r_radius * 2) * irisVisible

    # If the both eyes top and bottom distance is less than the minimum distance, then detection happened
    if ((leftDis < minEyeDis) and (rightDis < minEyeDis) and (not eyesClosed)):
        print(1)
        globals()["eyesClosed"] = True
    elif ((leftDis > minEyeDis) and (rightDis > minEyeDis) and (eyesClosed)):
        globals()["blinkCount"] = blinkCount + 1
        globals()["eyesClosed"] = False

    image = cv2.putText(image, "Blinks: " + str(blinkCount),
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
    return image


# Global and Const variables
# Mediapipe FaceMesh Object
mp_face_mesh = mp.solutions.face_mesh
FaceMesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# Eye indices List (For future references)
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390,
            249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154,
             155, 133, 173, 157, 158, 159, 160, 161, 246]

# Eyes top and bottom position indices
LEFT_EYE_TOP = 385
LEFT_EYE_BOTTOM = 374

RIGHT_EYE_TOP = 159
RIGHT_EYE_BOTTOM = 145

# Irises Indices list
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Iris shown %
irisVisible = 0.70  # 70%

# Frame size
FrameWidth, FrameHeight = 640, 360
# Window Name
windowName = "Blink Count"

# Initialize Camera
cap_vid = cv2.VideoCapture(1)
cap_vid.set(cv2.CAP_PROP_FRAME_WIDTH, FrameWidth)
cap_vid.set(cv2.CAP_PROP_FRAME_HEIGHT, FrameHeight)

blinkCount = 0
eyesClosed = False

print("\n[*] Blink Count Started...")
while True:
    ret, frame = cap_vid.read()
    if not ret:
        break

    frame = detectIrisAndCount(frame)

    # frame = cv2.flip(frame, cv2.CAP_PROP_XI_DECIMATION_HORIZONTAL)
    cv2.imshow(windowName, frame)

    if (cv2.waitKey(1) & 0xFF == 27):
        break
    if (cv2.getWindowProperty(windowName, cv2.WND_PROP_VISIBLE) < 1):
        break

# Release camera
cap_vid.release()


# Close all windows
print("\n[*] Blink Count complete with Count: " + str(blinkCount))
cv2.destroyAllWindows()
