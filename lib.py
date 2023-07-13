import cv2

def detect_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    #  eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Perform face detection using a pre-trained cascade classifier
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    #  eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    #  if len(faces) > 0 and len(eyes) > 1:
    if len(faces) > 0:
        cv2.rectangle(image, (faces[0][0], faces[0][1]), (faces[0][0] + faces[0][2], faces[0][1] + faces[0][3]), (0, 255, 0), 2)
        #  cv2.rectangle(image, (eyes[0][0], eyes[0][1]), (eyes[0][0] + eyes[0][2], eyes[0][1] + eyes[0][3]), (0, 255, 0), 2)
        #  cv2.rectangle(image, (eyes[1][0], eyes[1][1]), (eyes[1][0] + eyes[1][2], eyes[1][1] + eyes[1][3]), (0, 255, 0), 2)

    return image

