import cv2
import numpy as np
import mediapipe as mp

class DrowsinessDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1)

        
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.FACE_OUTLINE = list(range(0, 468))  

    def get_landmarks(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if results.multi_face_landmarks:
            return np.array([[p.x * w, p.y * h] for p in results.multi_face_landmarks[0].landmark])
        return None

    def calculate_ear(self, eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        return (A + B) / (2.0 * C)

    def get_ear(self, landmarks):
        left_eye = np.array([landmarks[i] for i in self.LEFT_EYE])
        right_eye = np.array([landmarks[i] for i in self.RIGHT_EYE])
        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)
        return (left_ear + right_ear) / 2.0

    def enhance_low_light(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_gray = clahe.apply(gray)

        
        frame = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

        
        avg_brightness = np.mean(gray)
        gamma = 2.0 if avg_brightness < 50 else 1.2  
        look_up_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
        frame = cv2.LUT(frame, look_up_table) 

        return frame
