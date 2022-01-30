import cv2
import mediapipe as mp


class FaceDetector:

    def __init__(self, mode=False, max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.5,
                 model_selection=0):
        self.results = None
        self.mode = mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.model_selection = model_selection
        self.mpFaceMesh = mp.solutions.face_mesh
        self.face = self.mpFaceMesh.FaceMesh(self.mode, self.max_num_faces, self.refine_landmarks,
                                             self.min_detection_confidence,
                                             self.model_selection)
        self.mpDraw = mp.solutions.drawing_utils
        self.connections = mp.solutions.face_mesh_connections
        self.drawSpec = self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)

    def detect_mesh(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face.process(img_rgb)
        if (self.results.multi_face_landmarks and draw):
            for face_landmarks in self.results.multi_face_landmarks:
                self.mpDraw.draw_landmarks(img, face_landmarks, self.connections.FACEMESH_TESSELATION, self.drawSpec,
                                           self.drawSpec)
        return img

    def find_position(self, img, draw=True):
        lm_list = []
        if (self.results.multi_face_landmarks):
            for face_landmarks in self.results.multi_face_landmarks:
                for id, lm in enumerate(face_landmarks.landmark):
                    h, w, c = img.shape
                    x, y = int(lm.x * w), int(lm.y * h)
                    lm_list.append((id, x, y))
                    if draw:
                        cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.6, (0, 255, 0))
        return lm_list
