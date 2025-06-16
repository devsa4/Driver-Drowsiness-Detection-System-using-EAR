import cv2
import time
import numpy as np
import pyttsx3
import winsound
import threading
from detector import DrowsinessDetector  

EAR_THRESHOLD = 0.25
ALARM_DURATION = 4

engine = pyttsx3.init()

def voice_alert():
    engine.say("Drowsiness detected! Please stay alert.")
    engine.runAndWait()

def beep_alert():
    winsound.Beep(1000, 1000)  

def alert():
    threading.Thread(target=voice_alert, daemon=True).start()
    threading.Thread(target=beep_alert, daemon=True).start()

eye_closed_start = None
alarm_on = False
ear_values = []  

detector = DrowsinessDetector()
cap = cv2.VideoCapture(0)
time.sleep(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    frame = detector.enhance_low_light(frame)

    landmarks = detector.get_landmarks(frame)

    if landmarks is not None:
        ear = detector.get_ear(landmarks)

        ear_values.append(ear)
        if len(ear_values) > 10:  # Maintain last 10 readings
            ear_values.pop(0)  
        avg_ear = sum(ear_values) / len(ear_values)  # Compute moving average

        detected_drowsy = avg_ear < EAR_THRESHOLD

        for index in detector.LEFT_EYE + detector.RIGHT_EYE:
            point = landmarks[index]
            cv2.circle(frame, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)  

        for index in detector.FACE_OUTLINE:
            point = landmarks[index]
            cv2.circle(frame, (int(point[0]), int(point[1])), 1, (255, 255, 255), -1)  

        cv2.putText(frame, f'Avg EAR: {avg_ear:.2f}', (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if detected_drowsy:
            if eye_closed_start is None:
                eye_closed_start = time.time()
            else:
                elapsed = time.time() - eye_closed_start
                if elapsed >= ALARM_DURATION:
                    if not alarm_on:
                        alarm_on = True
                        alert()  # Triggers voice + beep alert
                    cv2.putText(frame, "DROWSINESS ALERT!", (20, 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        else:
            eye_closed_start = None
            alarm_on = False

    cv2.imshow("Driver Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
