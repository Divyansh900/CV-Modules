import cv2 as cv
import mediapipe as mp
import time


def rescale(img, scale=1.0):
    h, w, c = img.shape
    dim = (int(w * scale), int(h * scale))
    return cv.resize(img, dim, interpolation=cv.INTER_LINEAR)


class faceMesh():

    def __init__(self, mode=False, faces=1, refine=False,
                 detection_confidence=0.5, tracking_confidence=0.5):
        self.mode = mode
        self.faces = faces
        self.refine = refine
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mpFace = mp.solutions.face_mesh
        self.face = self.mpFace.FaceMesh(max_num_faces=2)
        self.mpDraw = mp.solutions.drawing_utils
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

    def findFaceMesh(self, img, draw=True):

        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        lmlist = []
        faces = []

        self.results = self.face.process(imgRGB)
        # print(results.multi_face_landmarks)

        if self.results.multi_face_landmarks:
            for facelm in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, facelm, self.mpFace.FACEMESH_TESSELATION, self.drawSpec)

                for id, lm in enumerate(facelm.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmlist.append([cx, cy])

        faces.append(lmlist)
        return img, faces


def main():
    capture = cv.VideoCapture(1)
    ctime, ptime = 0, 0
    detector = faceMesh(faces=2)

    while True:
        success, img = capture.read()
        img, landmarks = detector.findFaceMesh(img)

        print(landmarks)

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        cv.putText(img, f"{int(fps)}", (10, 40), cv.FONT_HERSHEY_TRIPLEX,
                   1.0, (0, 180, 0), 2)
        img = rescale(img, 1.0)
        cv.imshow("Video", img)
        cv.waitKey(1)


if __name__ == "__main__":
    main()
