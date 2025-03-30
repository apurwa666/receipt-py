import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load model
model_path = 'app/models/ocr_model_v6.h5'  
model = load_model(model_path)

# Preprocess
def preprocess_character_image(char_img, target_size):
    # Resize
    char_img = cv2.resize(char_img, target_size)
    char_img = np.expand_dims(char_img, axis=-1)  
    char_img = np.expand_dims(char_img, axis=0)   
    
    # Normalize
    char_img = char_img / 255.0
    return char_img

# Predict
def predict_character(char_img):
    target_size = (28, 28) 
    img_array = preprocess_character_image(char_img, target_size)
    
    # Make prediction
    prediction = model.predict(img_array)
    
    predicted_class = np.argmax(prediction, axis=-1)
    return predicted_class

# Segment word
def segment_word_to_characters(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # Preprocess
    _, binary_img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    char_images = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter
        if w > 5 and h > 10: 
            char_image = binary_img[y:y+h, x:x+w]
            char_images.append(char_image)
    
    # Sort characters
    char_images = sorted(char_images, key=lambda x: cv2.boundingRect(x)[0])
    return char_images

# Main function
if __name__ == "__main__":
    img_path = 'C:/Users/apurw/OneDrive/Desktop/receipt_scanner_backend/test images/handword5.jpg' 
    
    
    character_images = segment_word_to_characters(img_path)
    
    
    predicted_word = []
    
    for char_image in character_images:
        
        predicted_class = predict_character(char_image)
        
        # Map the predicted class index 
        character_map = {0: 'W', 1: 'I', 2: 'S', 3: 'D', 4: 'O', 5: 'M'}  
        predicted_char = character_map.get(predicted_class[0], '?')  
        predicted_word.append(predicted_char)
    
    # Combine the predicted characters into a word
    print('Predicted Word:', ''.join(predicted_word))
    
  
    img = cv2.imread(img_path)
    cv2.putText(img, ''.join(predicted_word), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
