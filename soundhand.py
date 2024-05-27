import math
import cv2 as cv
import numpy as np
import HandTrackingmodel as htm
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Setup video capture from the default camera (usually webcam)
cap = cv.VideoCapture(0)
cap.set(3, 640)  # Set the width of the video frame
cap.set(4, 480)  # Set the height of the video frame

# Initialize the hand detector from the custom HandTrackingmodel module
detector = htm.HandDetector()

# Initialize audio utilities to control system volume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

# Get the volume range for the system volume
volrange = volume.GetVolumeRange()
minvol, maxvol = volrange[0], volrange[1]

# Initialize variables for volume control
vol = 0  # Current volume level
volb = 400  # Variable for drawing the volume bar
volp = 0  # Variable for displaying volume percentage

while True:
    success, img = cap.read()  # Read a frame from the camera
    img = detector.find_hands(img)  # Detect hands in the frame
    listlm = detector.find_position(img, draw=False)  # Get the positions of hand landmarks

    if len(listlm) != 0:
        # Get the coordinates of the tips of the thumb and index finger
        x1, y1 = listlm[4][1], listlm[4][2]
        x2, y2 = listlm[8][1], listlm[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Calculate the midpoint between the two tips

        # Draw circles at the tips of the thumb and index finger
        cv.circle(img, (x1, y1), 8, (255, 255, 255), cv.FILLED)
        cv.circle(img, (x2, y2), 8, (255, 255, 255), cv.FILLED)
        # Draw a line between the tips of the thumb and index finger
        cv.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
        # Draw a circle at the midpoint
        cv.circle(img, (cx, cy), 8, (100, 0, 255), cv.FILLED)

        # Calculate the distance between the tips of the thumb and index finger
        length = math.hypot(x2 - x1, y2 - y1)
        # Map the distance to the volume range
        vol = np.interp(length, [50, 300], [minvol, maxvol])
        # Map the distance to the volume bar height
        volb = np.interp(length, [50, 300], [400, 150])
        # Map the distance to the volume percentage
        volp = np.interp(length, [50, 300], [0, 100])

        print(vol)
        # Set the system volume to the calculated level
        volume.SetMasterVolumeLevel(vol, None)

        # If the distance is very small, change the color of the midpoint circle
        if length < 50:
            cv.circle(img, (cx, cy), 8, (0, 255, 255), cv.FILLED)

    # Draw the volume bar background
    cv.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    # Draw the current volume level on the volume bar
    cv.rectangle(img, (50, int(volb)), (85, 400), (255, 0, 255), cv.FILLED)
    # Display the volume percentage
    cv.putText(img, f'{int(volp)}%', (40, 450), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    # Show the frame with the drawings
    cv.imshow("Image", img)
    # Break the loop if 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv.destroyAllWindows()
