import cv2
import mediapipe as mp


class FaceDetector:

    def __init__(self, min_detection_confidence=0.5, model_selection=0):
        self.results = None
        self.min_detection_confidence = min_detection_confidence
        self.model_selection = model_selection
        self.mpFaceDetection = mp.solutions.face_detection
        self.face = self.mpFaceDetection.FaceDetection(min_detection_confidence, model_selection)
        self.mpDraw = mp.solutions.drawing_utils

    def detect_face(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face.process(img_rgb)
        if self.results.detections and draw:
            for detection in self.results.detections:
                self.mpDraw.draw_detection(img, detection)
        return img

    def find_position(self, img, draw=True):
        bboxes = []
        if self.results.detections:
            for detection in self.results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, c = img.shape
                bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                if draw:
                    cv2.rectangle(img, bbox, (255, 0, 255), 2)
                    cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                                2,
                                (255, 0, 255), 2)
                bboxes.append(bbox)
        return bboxes
