import cv2
import pickle
import numpy as np
import os
import face_recognition
from sklearn.metrics.pairwise import cosine_similarity

# Set a threshold for recognizing someone
SIMILARITY_THRESHOLD = 0.97

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")

# Load stored face embeddings and labels
with open('data/names.pkl', 'rb') as f:
    LABELS = pickle.load(f)

with open('data/face_embeddings.pkl', 'rb') as f:
    EMBEDDINGS = pickle.load(f)

while True:
    ret, frame = video.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_locations(rgb_frame)

    for (top, right, bottom, left) in faces:
        face_encodings = face_recognition.face_encodings(rgb_frame, [(top, right, bottom, left)])
        
        if face_encodings:
            current_embedding = face_encodings[0].reshape(1, -1)
            
            # Compute cosine similarity between the current face and stored faces
            similarities = cosine_similarity(current_embedding, EMBEDDINGS)
            max_similarity_index = np.argmax(similarities)
            max_similarity = similarities[0][max_similarity_index]
            
            if max_similarity > SIMILARITY_THRESHOLD:
                name = LABELS[max_similarity_index]
            else:
                name = "Unknown"
                
            # Display the result
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, top - 40), (right, top), (50, 50, 255), -1)
            cv2.putText(frame, name, (left, top - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

    cv2.imshow("Frame", frame)
    
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
