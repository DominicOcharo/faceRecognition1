import cv2
import pickle
import numpy as np
import os
import face_recognition  # Using face_recognition for extracting face embeddings

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")

name = input("Enter Your Name: ")

embeddings = []

while True:
    ret, frame = video.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_locations(rgb_frame)

    for (top, right, bottom, left) in faces:
        face_encodings = face_recognition.face_encodings(rgb_frame, [(top, right, bottom, left)])
        
        if face_encodings:
            embeddings.append(face_encodings[0])
            cv2.putText(frame, str(len(embeddings)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
            cv2.rectangle(frame, (left, top), (right, bottom), (50, 50, 255), 2)
    
    cv2.imshow("Frame", frame)
    
    k = cv2.waitKey(1)
    if k == ord('q') or len(embeddings) == 100:
        break

video.release()
cv2.destroyAllWindows()

embeddings = np.asarray(embeddings)

# Saving face embeddings and labels
if 'names.pkl' not in os.listdir('data/'):
    names = [name] * len(embeddings)
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    with open('data/names.pkl', 'rb') as f:
        names = pickle.load(f)
    names += [name] * len(embeddings)
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)

if 'face_embeddings.pkl' not in os.listdir('data/'):
    with open('data/face_embeddings.pkl', 'wb') as f:
        pickle.dump(embeddings, f)
else:
    with open('data/face_embeddings.pkl', 'rb') as f:
        face_embeddings = pickle.load(f)
    face_embeddings = np.append(face_embeddings, embeddings, axis=0)
    with open('data/face_embeddings.pkl', 'wb') as f:
        pickle.dump(face_embeddings, f)
