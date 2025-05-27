# Driver-Drowsiness-Detection-System-using-EAR
Driver drowsiness is a major cause of road accidents worldwide, often resulting in severe consequences due to delayed human response times. This project aims to develop a real-time Driver Drowsiness Detection System that alerts the driver when signs of fatigue are detected, potentially reducing the risk of accidents.

The primary objective is to reduce accidents caused by drowsy driving, which is a significant factor in road fatalities. The system can be integrated into vehicles as a built-in safety feature or implemented as a standalone device, making it accessible for various applications, including commercial transportation and personal vehicles. The system uses a standard webcam to monitor the driver’s facial landmarks in real-time using Media Pipe, a lightweight and efficient framework for face detection. Specifically, the system tracks the eyes and calculates the Eye Aspect Ratio (EAR) to detect whether the driver’s eyes remain closed for an extended period. If the eyes remain closed for more than 4 seconds, the system triggers an audible alarm using Windows’ native sound system (winsound) to wake the driver.
The project leverages computer vision and deep learning to detect driver drowsiness in real time. It integrates face landmark detection, eye aspect ratio (EAR) analysis, and a pre-trained deep learning model to improve accuracy and responsiveness. The system continuously monitors eye closure duration and triggers an alarm if the driver’s eyes remain closed beyond a predefined threshold.
Architecture Overview
1.	Data Acquisition & Processing
o	The system captures video frames using OpenCV and CLAHE
o	Converts frames to grayscale for efficient processing.
2.	Face & Eye Detection
o	Uses facial landmark detector to extract key facial points.
o	Computes the EAR based on eye landmarks to detect drowsiness patterns.
3.	Drowsiness Alert Mechanism
o	If EAR falls below the threshold (0.25) for few seconds, an audio alert is triggered using winsound.
o	Real-time feedback is displayed on-screen with status updates..
