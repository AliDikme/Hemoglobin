import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.models import Model
from mtcnn import MTCNN

# Initialize face detector
detector = MTCNN()

# Load the pre-trained MobileNet model
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Model for feature extraction
model = Model(inputs=base_model.input, outputs=base_model.output)

# Initialize an empty list to hold the features
features = []

# Open the video file
video = cv2.VideoCapture('D:\Hemoglobin\Features_Extractor_Test\Halise genek - 11.9.mp4')

# Process the video
while True:
    # Read the next frame
    ret, frame = video.read()

    # If the frame was not read correctly, then we have reached the end of the video
    if not ret:
        break

    # Detect faces in the frame
    result = detector.detect_faces(frame)

    # If a face is detected
    if result:
        # Get the bounding box for the first face
        x1, y1, width, height = result[0]['box']
        x2, y2 = x1 + width, y1 + height

        # Extract the face from the frame
        face = np.array(frame[y1:y2, x1:x2])

        # Resize the face to 224x224 (the size expected by MobileNet)
        face = cv2.resize(face, (224, 224))

        # Preprocess the face image
        face = preprocess_input(face)

        # Add an extra dimension
        face = np.expand_dims(face, axis=0)

        # Extract features using the MobileNet model
        feature = model.predict(face)

        # Remove single-dimensional entries from the shape of the array
        feature = np.squeeze(feature)

        # Append the feature to the list of features
        features.append(feature)

# Release the video file
video.release()
