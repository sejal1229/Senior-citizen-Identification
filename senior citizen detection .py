import tkinter as tk
from tkinter import filedialog
import cv2
import tensorflow as tf
import numpy as np
from datetime import datetime
import pandas as pd

# Load saved model
model = tf.keras.models.load_model('Age_Sex_Detection.keras')

# Initializing CSV file
csv_file = 'results.csv'
with open(csv_file, 'w') as file:
    file.write('Age,Gender,Senior Citizen,Time of Visit\n')

def preprocess_face(face):
    face = cv2.resize(face, (48, 48))  
    face = face.astype('float32') / 255.0  # Normalize
    face = np.expand_dims(face, axis=0)  
    return face

def predict_age_gender(face):
    face = preprocess_face(face)
    predictions = model.predict(face)
    print(predictions) 
    age = int(np.round(predictions[1][0]))
    gender_prob = predictions[0][0][0]
    gender = 'Male' if gender_prob > 0.5 else 'Female' 
    return age, gender


def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def process_video(file_path):
    cap = cv2.VideoCapture(file_path)  
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break
        
        faces = detect_faces(frame)
        print(f"Detected faces: {faces}")  

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            age, gender = predict_age_gender(face)
            senior_citizen = age > 60
            senior_status = 'Yes' if senior_citizen else 'No'
        
            # Debug: Print data before writing
            print(f"Writing to CSV: Age={age}, Gender={gender}, Senior Citizen={senior_status}")

             # Save results to CSV
            time_of_visit = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(csv_file, 'a') as file:
               file.write(f"{age},{gender},{senior_status},{time_of_visit}\n")

                # Draw rectangle and text on frame
               cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
               cv2.putText(frame, f'Age: {age}, Gender: {gender}, Senior Citizen: {senior_status}', 
                          (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
         # Add exit instruction
        cv2.putText(frame, 'Press "q" to exit', (10, frame.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('Video Feed', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def upload_video():
    file_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4;*.avi")])
    print(f"Selected file: {file_path}")  # Debug: Print selected file path
    if file_path:
        process_video(file_path)
# GUI Setup
root = tk.Tk()
root.title("Video Age and Gender Detection")
upload_button = tk.Button(root, text="Upload Video", command=upload_video)
upload_button.pack(padx=20, pady=20)

# Run the GUI loop
root.mainloop()
