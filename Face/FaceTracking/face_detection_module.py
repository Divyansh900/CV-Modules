import cv2 as cv
import mediapipe as mp
import time


def rescale(image, scale=1.0):
    dim = (int(image.shape[1] * scale),
           int(image.shape[0] * scale))
    return cv.resize(image, dim, interpolation=cv.INTER_LINEAR)

    # print(results.detections)


class faceDetector():

    def __init__(self, confidence=0.5, model=0):
        self.confidence = confidence
        self.model = model

        self.mpFace = mp.solutions.face_detection
        self.face = self.mpFace.FaceDetection()
        self.mpDraw = mp.solutions.drawing_utils

    def findFace(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        self.results = self.face.process(imgRGB)

        bboxs = []

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bbox_ = detection.location_data.relative_bounding_box
                # mpDraw.draw_detection(img, detection)

                h, w, c = img.shape
                bbox = int(bbox_.xmin * w), int(bbox_.ymin * h), \
                    int(bbox_.width * w), int(bbox_.height * h)

                bboxs.append([id, bbox, detection.score])

                cv.rectangle(img, bbox, (55, 255, 40), 2)

                if draw:
                    acc_text = f"Confidence : {int(detection.score[0] * 100)}%"
                    cv.putText(img, acc_text,
                               (bbox[0], bbox[1] - 6), cv.FONT_HERSHEY_TRIPLEX,
                               0.5, (150, 250, 255), 1)

        return img, bboxs


def main():
    capture = cv.VideoCapture("video (2160p)(1).mp4")

    cTime, pTime = 0, 0

    detector = faceDetector()

    while True:
        success, img = capture.read()
        img = rescale(img, scale=0.2)
        img = cv.flip(img, 1)
        img, bboxs = detector.findFace(img)
        print(bboxs)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv.putText(img, str(int(fps)),
                   (10, 50), cv.FONT_HERSHEY_TRIPLEX,
                   1.0, (150, 250, 55), 2)

        cv.imshow("Video", img)
        cv.waitKey(1)


if __name__ == "__main__":
    main()
