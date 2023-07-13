import cv2
from fer import FER

detector = FER()

def detect_face(image):
    face_info = detector.detect_emotions(image)
    if face_info:
        bounding_box = face_info[0]["box"]
        cv2.rectangle(image, (
            bounding_box[0], bounding_box[1]),
            (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
            (0, 155, 255), 2)
    return image
