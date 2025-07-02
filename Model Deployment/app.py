import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow import keras

model = keras.models.load_model('CNN_model.h5')
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
detector = MTCNN()

def preprocess_fer2013(img):
    img = img.astype(np.uint8)
    img = cv2.fastNlMeansDenoising(img, None, h=7, templateWindowSize=5, searchWindowSize=15)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    blurred = cv2.GaussianBlur(img, (3, 3), 1.0)
    img = cv2.addWeighted(img, 1.2, blurred, -0.2, 0)
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    return img

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(frame_rgb)

    for face in faces:
        x, y, w, h = face['box']
        x, y = max(0, x), max(0, y)
        face_crop = frame[y:y+h, x:x+w]

        try:
            face_gray = cv2.cvtColor(cv2.resize(face_crop, (48, 48)), cv2.COLOR_BGR2GRAY)
            face_proc = preprocess_fer2013(face_gray)
            face_input = np.expand_dims(face_proc, axis=0)

            pred = model.predict(face_input)[0]
            emotion_idx = np.argmax(pred)
            emotion_label = class_names[emotion_idx]
            confidence = np.max(pred)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(frame, f"{emotion_label} ({confidence*100:.1f}%)",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        except Exception as e:
            print("Face processing error:", e)

    cv2.imshow('Live Emotion Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
