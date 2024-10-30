import cv2
import pickle
import numpy as np
import os
import face_recognition
from fastapi import FastAPI, HTTPException, Form
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
import asyncio

app = FastAPI()

# Path to the data folder
DATA_PATH = "./data"

# Function to save face embeddings and names
def save_face_data(name: str, embeddings: np.ndarray):
    # Save names
    if 'names.pkl' not in os.listdir(DATA_PATH):
        names = [name] * len(embeddings)
        with open(f'{DATA_PATH}/names.pkl', 'wb') as f:
            pickle.dump(names, f)
    else:
        with open(f'{DATA_PATH}/names.pkl', 'rb') as f:
            names = pickle.load(f)
        names += [name] * len(embeddings)
        with open(f'{DATA_PATH}/names.pkl', 'wb') as f:
            pickle.dump(names, f)

    # Save embeddings
    if 'face_embeddings.pkl' not in os.listdir(DATA_PATH):
        with open(f'{DATA_PATH}/face_embeddings.pkl', 'wb') as f:
            pickle.dump(embeddings, f)
    else:
        with open(f'{DATA_PATH}/face_embeddings.pkl', 'rb') as f:
            face_embeddings = pickle.load(f)
        face_embeddings = np.append(face_embeddings, embeddings, axis=0)
        with open(f'{DATA_PATH}/face_embeddings.pkl', 'wb') as f:
            pickle.dump(face_embeddings, f)


@app.post("/register/")
async def register_user(name: str = Form(...)):
    video = cv2.VideoCapture(0)
    facedetect = cv2.CascadeClassifier(f"{DATA_PATH}/haarcascade_frontalface_default.xml")
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

        if cv2.waitKey(1) == ord('q') or len(embeddings) == 100:
            break

    video.release()
    cv2.destroyAllWindows()

    embeddings = np.asarray(embeddings)
    save_face_data(name, embeddings)

    return {"message": "User registered successfully!"}


@app.post("/login/")
async def login_user():
    SIMILARITY_THRESHOLD = 0.97
    timeout = 15  # Max time for detection

    video = cv2.VideoCapture(0)
    facedetect = cv2.CascadeClassifier(f"{DATA_PATH}/haarcascade_frontalface_default.xml")

    # Load stored face embeddings and labels
    if 'names.pkl' not in os.listdir(DATA_PATH) or 'face_embeddings.pkl' not in os.listdir(DATA_PATH):
        raise HTTPException(status_code=404, detail="No registered users. Please register first.")

    with open(f'{DATA_PATH}/names.pkl', 'rb') as f:
        LABELS = pickle.load(f)

    with open(f'{DATA_PATH}/face_embeddings.pkl', 'rb') as f:
        EMBEDDINGS = pickle.load(f)

    start_time = cv2.getTickCount()
    while (cv2.getTickCount() - start_time) / cv2.getTickFrequency() < timeout:
        ret, frame = video.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_locations(rgb_frame)

        for (top, right, bottom, left) in faces:
            face_encodings = face_recognition.face_encodings(rgb_frame, [(top, right, bottom, left)])
            if face_encodings:
                current_embedding = face_encodings[0].reshape(1, -1)
                similarities = cosine_similarity(current_embedding, EMBEDDINGS)
                max_similarity_index = np.argmax(similarities)
                max_similarity = similarities[0][max_similarity_index]

                if max_similarity > SIMILARITY_THRESHOLD:
                    name = LABELS[max_similarity_index]
                    video.release()
                    cv2.destroyAllWindows()
                    return {"message": f"Login successful! Welcome, {name}."}

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
    raise HTTPException(status_code=401, detail="Login unsuccessful. Please try again or register.")

