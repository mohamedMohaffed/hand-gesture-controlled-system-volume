import cv2 as cv
import mediapipe as mp
import time

class HandDetector:
    def __init__(self, mode=False, max_hands=2, detection_conf=0.5, track_conf=0.5):
        # Constructor to initialize hand detection parameters
        self.mode = mode
        self.max_hands = max_hands
        self.detection_conf = detection_conf
        self.track_conf = track_conf
        self.mphand = mp.solutions.hands
        # Initialize the Hand module from the MediaPipe library with specified parameters
        self.hands = self.mphand.Hands(static_image_mode=self.mode,
                                       max_num_hands=self.max_hands,
                                       min_detection_confidence=self.detection_conf,
                                       min_tracking_confidence=self.track_conf)
        self.mpdraw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        # Method to detect hands in the input image
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        # Process the image and get hand landmarks
        self.res = self.hands.process(imgRGB)
        if self.res.multi_hand_landmarks:
            for hand_lms in self.res.multi_hand_landmarks:
                if draw:
                    # Draw hand landmarks and connections on the image
                    self.mpdraw.draw_landmarks(img, hand_lms, self.mphand.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_no=0, draw=True):
        # Method to find the landmarks of a specific hand in the input image
        lm_list = []
        if self.res.multi_hand_landmarks:
            my_hand = self.res.multi_hand_landmarks[hand_no]
            for id, lm in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((id, cx, cy))
                if draw:
                    # Draw a circle on the tip of the index finger
                    if id == 8:
                        cv.circle(img, (cx, cy), 10, (255, 255, 0), cv.FILLED)
        return lm_list

def main():
    ptime = 0
    ctime = 0
    # Capture video from the default camera
    cap = cv.VideoCapture(0)
    # Create a HandDetector object
    detector = HandDetector()

    while True:
        # Read a frame from the camera
        success, img = cap.read()
        if not success:
            break

        # Find and draw hands in the frame
        img = detector.find_hands(img)
        # Find hand landmarks and positions
        lm_list = detector.find_position(img)

        # Calculate and display the frame rate
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv.putText(img, f'FPS: {int(fps)}', (20, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 2)

        # Display the image
        cv.imshow("Image", img)

        # Exit the loop if 'Esc' key is pressed
        if cv.waitKey(1) & 0xFF == 27:
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
