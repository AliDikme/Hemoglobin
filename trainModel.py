import os
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.models import Model
from mtcnn import MTCNN
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Flatten
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# Initialize face detector
detector = MTCNN()

# Load the pre-trained MobileNet model
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Model for feature extraction
model = Model(inputs=base_model.input, outputs=base_model.output)


features = []
labels = []

# Directory where the videos are located
video_dir = 'D:\Hemoglobin\PreprocessedFrames'
counter = 0
# Loop over each file in the directory
def trainModelTensorFlow(video_dir):
    # Initialize face detector
    detector = MTCNN()

    # Load the pre-trained MobileNet model
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Model for feature extraction
    model = Model(inputs=base_model.input, outputs=base_model.output)

    features = []
    labels = []
    counter = 0

    for filename in os.listdir(video_dir):
        hemoglobin = float(filename.split('_')[1].split('.mp4')[0])
        video_path = os.path.join(video_dir, filename)
        video = cv2.VideoCapture(video_path)

        while True:
            ret, frame = video.read()
            if not ret:
                break

            result = detector.detect_faces(frame)
            if result:
                x1, y1, width, height = result[0]['box']
                x2, y2 = x1 + width, y1 + height
                face = np.array(frame[y1:y2, x1:x2])
                face = cv2.resize(face, (224, 224))
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)
                feature = model.predict(face)
                feature = np.squeeze(feature)
                features.append(feature)
                labels.append(hemoglobin)

        video.release()
        counter += 1
        print("This is counter video : " + str(counter))

    features = np.array(features)
    labels = np.array(labels)
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2,
                                                                                random_state=42)

    model = Sequential([
        Flatten(),
        Dense(128, activation='relu', input_shape=(7 * 7 * 1024,)),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    history = model.fit(features_train, labels_train, validation_data=(features_test, labels_test), epochs=10,
                        batch_size=32)

    # predict values
    predictions = model.predict(features_test)
    rms = mean_squared_error(labels_test, predictions,
                             squared=False)  # if squared=True, this returns MSE; if False, returns RMSE

    print("Root Mean Square Error: ", rms)

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    model.save('prediction_hemoglobin_rms.h5')

trainModelTensorFlow(video_dir)

def extractFeaturesFromVideo(videoPath):
    # Initialize an empty list to hold the features
    new_features = []

    # Open the video file
    video = cv2.VideoCapture(videoPath)

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
            new_features.append(feature)

    # Release the video file
    video.release()

    # Convert your list of features to a numpy array
    new_features_array = np.array(new_features)

    # Average the features over all frames
    avg_features = np.mean(new_features_array, axis=0)

    return avg_features

def extractFeaturesFromFrames(frames):
    # Initialize an empty list to hold the features
    new_features = []

    # Process each frame
    for frame in frames:
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
            new_features.append(feature)

    # Convert your list of features to a numpy array
    new_features_array = np.array(new_features)

    # Average the features over all frames
    avg_features = np.mean(new_features_array, axis=0)

    return avg_features

    # Initialize lists to hold the features and labels

