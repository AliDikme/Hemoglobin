import numpy as np
import trainModel
# Load the model
from tensorflow.keras.models import load_model
model = load_model('my_model.h5')
videoPath = 'Features_Extractor_Test/Nilüfer Günelshorten_11.2.mp4'

# Ensure new_features is a numpy array and add an extra dimension
# Extract features from a new video
new_features = trainModel.extractFeaturesFromVideo(videoPath)

# Add an extra dimension to the input data
new_features = np.expand_dims(new_features, axis=0)

# Predict the hemoglobin level
predicted_hemoglobin = model.predict(new_features)

# The output will be a 2D array with one value, you can extract it using
predicted_hemoglobin = predicted_hemoglobin[0][0]


print("prediction result for demo : ")
print(predicted_hemoglobin)