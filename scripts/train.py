import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Function to load images from .ubyte files
def load_images_from_ubyte(file_path):
    with open(file_path, 'rb') as f:
        # Read header information
        magic_number = int.from_bytes(f.read(4), 'big')  # Should be 2051 for images
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')

        # Read the image data
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows, cols, 1)  # Reshape to (num_images, rows, cols, channels)
    return images

# Function to load labels from .ubyte files
def load_labels_from_ubyte(file_path):
    with open(file_path, 'rb') as f:
        # Read header information
        magic_number = int.from_bytes(f.read(4), 'big')  # Should be 2049 for labels
        num_labels = int.from_bytes(f.read(4), 'big')

        # Read the label data
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# Load training and testing data
train_images = load_images_from_ubyte('C:/Users/apurw/OneDrive/Desktop/Alt_OCR/react-/dataset/train-images.idx3-ubyte')
train_labels = load_labels_from_ubyte('C:/Users/apurw/OneDrive/Desktop/Alt_OCR/react-/dataset/train-labels.idx1-ubyte')

test_images = load_images_from_ubyte('C:/Users/apurw/OneDrive/Desktop/Alt_OCR/react-/dataset/t10k-images.idx3-ubyte')
test_labels = load_labels_from_ubyte('C:/Users/apurw/OneDrive/Desktop/Alt_OCR/react-/dataset/t10k-labels.idx1-ubyte')

# Normalize the image data to range [0, 1]
train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0

# Convert labels to categorical (one-hot encoding)
num_classes = len(np.unique(train_labels))  # Total number of unique classes
train_labels = to_categorical(train_labels, num_classes=num_classes)
test_labels = to_categorical(test_labels, num_classes=num_classes)

# Split the train data further into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.1, random_state=42)

# Ensure that the input shape is compatible with the model (28, 28, 1)
print(f"Train images shape: {X_train.shape}")
print(f"Validation images shape: {X_val.shape}")
print(f"Test images shape: {test_images.shape}")

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # Assuming 28x28 images
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(128, (3, 3), activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
batch_size = 64
epochs = 100

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=batch_size, epochs=epochs)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')

# Save the trained model
model.save('app/models/ocr_model_v6.h5')
