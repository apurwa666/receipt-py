import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

MODEL_PATH = 'app/models/ocr_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)

def extract_text(image_path):
    image = load_img(image_path, target_size=(32, 32), color_mode='grayscale')
    image = img_to_array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    label_index = np.argmax(prediction)
    
    # Map predicted label index to actual label (adjust based on your model's training data)
    index_to_label = {0: "name", 1: "item", 2: "price", 3: "date"}  # Example mapping
    return index_to_label.get(label_index, "unknown")
