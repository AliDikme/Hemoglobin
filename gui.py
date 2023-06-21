from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
import numpy as np
import trainModel
from tensorflow.keras.models import load_model

class HemoglobinApp(App):
    def build(self):
        self.camera = cv2.VideoCapture(0)
        self.frames = [] # Added to hold frames for processing
        self.processing = False # Added to indicate if processing is on

        layout = BoxLayout(orientation='vertical')

        # Add the camera view
        self.image = Image()
        layout.add_widget(self.image)

        # Add the button
        self.button = Button(text='Start Processing')
        self.button.on_release = self.start_processing
        layout.add_widget(self.button)

        # Add the hemoglobin label
        self.label = Label(text="Hemoglobin: 0.0")
        layout.add_widget(self.label)

        # Schedule the update function to be called every frame
        Clock.schedule_interval(self.update, 1.0 / 30.0) # Adjust to your camera's FPS

        return layout

    def update(self, dt):
        ret, frame = self.camera.read()

        # Make sure the frame was read correctly
        if not ret:
            return

        # If processing, add frame to frames list
        if self.processing:
            self.frames.append(frame)

        # Convert the frame to texture
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tostring()
        texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

        # Display the image in the Image widget
        self.image.texture = texture1

    def start_processing(self):
        # Start processing frames
        self.processing = True

        # After 3 seconds, stop processing and predict
        Clock.schedule_once(self.stop_processing, 3)

    def stop_processing(self, dt):
        # Stop processing frames
        self.processing = False

        # Call your feature extraction function
        new_features = trainModel.extractFeaturesFromFrames(self.frames)
        model = load_model('prediction_hemoglobin.h5')
        # TODO: Use the features to predict the hemoglobin level
        # new_hemoglobin = model.predict(new_features)
        # Add an extra dimension to the input data
        new_features = np.expand_dims(new_features, axis=0)

        # Predict the hemoglobin level
        predicted_hemoglobin = model.predict(new_features)

        # The output will be a 2D array with one value, you can extract it using
        predicted_hemoglobin = predicted_hemoglobin[0][0]

        print("prediction result for demo : ")
        print(predicted_hemoglobin)

        # Update the hemoglobin value
        self.label.text = f"Hemoglobin: {predicted_hemoglobin:.2f}"

        # Clear frames list
        self.frames = []

HemoglobinApp().run()
