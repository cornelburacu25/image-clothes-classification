import tensorflow as tf
import numpy as np
import cv2

# Load the trained model
model = tf.keras.models.load_model('vgg16Model.keras')

# Load new data for prediction
image = cv2.imread('fusta.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224, 224))
data = np.array([image]) / 255.0

# Get the predicted probabilities for each class
probabilities = model.predict(data)

# Get the predicted class index (i.e., the class with the highest probability)
predicted_class = np.argmax(probabilities)

# Print the predicted class and probabilities
class_names = ['pants', 't-shirt', 'skirt', 'dress', 'shorts', 'shoes', 'hat', 'longsleeve', 'outwear', 'shirt']
print('Predicted class:', class_names[predicted_class])
print('Probabilities:', probabilities)
