import cv2
import cv2 as cv
import mediapipe as mp
import time


class poseDetector():

    def __init__(self, mode=False, complexity=1, smooth_lm=True,
                 segmentation=False, smooth_segments=True,
                 detection_confidence=0.5, tracking_confidence=0.5):
        self.mode = mode
        self.complexity = complexity
        self.smooth_lm = smooth_lm
        self.segmentation = segmentation
        self.smooth_segments = smooth_segments
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(mode, complexity, smooth_lm, segmentation,
                                     smooth_segments, detection_confidence, tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        lmlist = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                lmlist.append([id, cx, cy])

                if draw:
                    cv.circle(img, (cx, cy), 5,
                              (120, 80, 255), thickness=-1)
        return lmlist


def main():
    capture = cv.VideoCapture("./Video13.mp4")

    ctime, ptime = 0, 0

    detector = poseDetector()

    while True:
        success, img = capture.read()

        img = detector.findPose(img)

        lmlist = detector.findPosition(img)
        print(lmlist)

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        cv.putText(img, str(int(fps)), (10, 30),
                   cv2.FONT_HERSHEY_TRIPLEX, 1,
                   (150, 120, 120), 1)

        cv.imshow("Video Stream", img)
        cv.waitKey(1)


if __name__ == "__main__":
    main()
