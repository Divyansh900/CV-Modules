import cv2 as cv
import mediapipe as mp
import time


class hand_detector():

    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, track_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.track_confidence = track_confidence

        # Module for hand detection
        self.mpHands = mp.solutions.hands

        # Object
        self.hands = self.mpHands.Hands(self.mode, self.max_hands, 1,
                                        self.detection_confidence, self.track_confidence)

        # Class for visualization
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True, highlight=True):

        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:

                if draw == True:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):

        lmlist = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape

                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])

                if draw:
                    if id == 4 or id == 8 or id == 12 or id == 16 or id == 20:
                        cv.circle(img, (cx, cy), 11,
                                  (120, 255, 150), thickness=-1)
        return lmlist


# dummy code for running the module

# pTime = 0
# cTime = 0
# capture = cv.VideoCapture(1)
#
# detector = hand_detector()
#
# while True:
#     success, img = capture.read()
#     img = detector.findHands(img)
#     lmList = detector.findPosition(img)
#     if len(lmList) != 0:
#         print(lmList[4])
#
#     cTime = time.time()
#     fps = 1 / (cTime - pTime)
#     pTime = cTime
#
#     img = cv.flip(img, 1)
#     cv.putText(img, str(int(fps)), (10, 70),
#                cv.FONT_HERSHEY_TRIPLEX, 2, (250, 250, 250),
#                3)
#
#     cv.imshow("Image", img)
#
#     cv.waitKey(1)


def main():
    pTime = 0
    cTime = 0
    capture = cv.VideoCapture(1)

    detector = hand_detector()

    while True:
        success, img = capture.read()
        img = detector.findHands(img, draw=False)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        img = cv.flip(img, 1)
        cv.putText(img, str(int(fps)), (10, 70),
                   cv.FONT_HERSHEY_TRIPLEX, 2, (250, 250, 250),
                   3)

        cv.imshow("Image", img)

        cv.waitKey(1)


if __name__ == "__main__":
    main()
