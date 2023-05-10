import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential

# Path to the directory containing training samples
training_dir = "./class"

# Path to the directory containing input images for classification
input_dir = "./input"

# Define the model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))

# Load training samples
training_samples = {}
class_to_label = {}
label_to_class = {}
label_counter = 0

for class_name in os.listdir(training_dir):
    class_samples = []
    class_path = os.path.join(training_dir, class_name)
    if os.path.isdir(class_path):
        class_to_label[class_name] = label_counter
        label_to_class[label_counter] = class_name
        for file_name in os.listdir(class_path):
            image_path = os.path.join(class_path, file_name)
            image = load_img(image_path, target_size=(224, 224))
            image = img_to_array(image)
            image = image / 255.0  # Normalize the image pixels
            class_samples.append(image)
        training_samples[label_counter] = class_samples
        label_counter += 1

num_classes = len(class_to_label)

# Train the model
X_train = []
y_train = []
for label, samples in training_samples.items():
    X_train.extend(samples)
    y_train.extend([label] * len(samples))

X_train = np.array(X_train)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)

model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20)

# Classify input images
threshold = 0.65  # Set the probability threshold for detection

for file_name in os.listdir(input_dir):
    image_path = os.path.join(input_dir, file_name)
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image / 255.0  # Normalize the image pixels
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image)
    predicted_label = np.argmax(predictions)
    predicted_probability = np.max(predictions)
    if predicted_probability < threshold:
        predicted_class = 'Not detected'
    else:
        predicted_class = label_to_class[predicted_label]

    print(f"{file_name}: {predicted_class} ({predicted_probability})")
