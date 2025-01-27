import json
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import math

# Function to detect symmetry lines
def detect_symmetry_lines(image):
    rows, cols = image.shape
    center_x, center_y = cols // 2, rows // 2

    horizontal_symmetry = np.all(image[:center_y, :] == image[center_y:, :][::-1, :])
    vertical_symmetry = np.all(image[:, :center_x] == image[:, center_x:][:, ::-1])
    
    diagonal1_symmetry = np.all(image[:center_y, :center_x] == image[center_y:, center_x:][::-1, ::-1])
    diagonal2_symmetry = np.all(np.fliplr(image[:center_y, :center_x]) == image[center_y:, center_x:][::-1, ::-1])
    
    return horizontal_symmetry, vertical_symmetry, diagonal1_symmetry, diagonal2_symmetry

# Function to extract features from an image
def extract_features(image):
    h_sym, v_sym, d1_sym, d2_sym = detect_symmetry_lines(image)
    symmetry_count = sum([h_sym, v_sym, d1_sym, d2_sym])
    
    # Flatten image for additional features
    flattened_image = image.flatten()
    
    # Count edges to estimate the number of sides
    edges = cv2.Canny(image, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    num_sides = 0
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.04 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        num_sides = len(approx)
    
    # Calculate angles of the sides
    angles = []
    for i in range(len(approx)):
        p1 = approx[i][0]
        p2 = approx[(i + 1) % len(approx)][0]
        p3 = approx[(i + 2) % len(approx)][0]
        
        v1 = p1 - p2
        v2 = p3 - p2
        angle = math.degrees(math.acos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
        angles.append(angle)
    
    mean_angle = np.mean(angles) if angles else 0
    std_angle = np.std(angles) if angles else 0
    
    # Feature vector
    features = np.array([
        symmetry_count,
        h_sym,
        v_sym,
        d1_sym,
        d2_sym,
        num_sides,
        mean_angle,
        std_angle,
        np.mean(flattened_image),
        np.std(flattened_image)
    ])
    
    return features

# Function to convert strokes to an image
def strokes_to_image(strokes, size=(32, 32)):
    image = np.zeros(size, dtype=np.uint8)
    for stroke in strokes:
        for i in range(len(stroke[0]) - 1):
            x1, y1 = int(stroke[0][i]), int(stroke[1][i])
            x2, y2 = int(stroke[0][i+1]), int(stroke[1][i+1])
            cv2.line(image, (x1, y1), (x2, y2), 255, 1)
    return image

# Function to preprocess image data
def preprocess_image_data(file_path, label):
    images = []
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            strokes = data['drawing']
            image = strokes_to_image(strokes, size=(32, 32))
            image_rgb = np.stack([image] * 3, axis=-1)  # Convert grayscale to RGB
            images.append(image_rgb)
            labels.append(label)
    return np.array(images), np.array(labels)

# Function to build CNN model
def build_cnn_model(input_shape=(32, 32, 3)):
    # Load the VGG16 model, excluding the top layers
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze the VGG16 model

    # Create a functional model
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(2, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Main function to execute the code
def main():
    circle_file = r'/Users/lakshitajawandhiya/Desktop/auto-draw-clone/AdobeGenSolve/data/quickdraw-dataset/examples/full_simplified_circle.ndjson'
    triangle_file = r'/Users/lakshitajawandhiya/Desktop/auto-draw-clone/AdobeGenSolve/data/quickdraw-dataset/examples/full_simplified_triangle.ndjson'
    
    circle_images, circle_labels = preprocess_image_data(circle_file, 0)
    triangle_images, triangle_labels = preprocess_image_data(triangle_file, 1)
    
    images = np.concatenate([circle_images, triangle_images], axis=0)
    labels = np.concatenate([circle_labels, triangle_labels], axis=0)
    
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    X_train = X_train / 255.0
    X_val = X_val / 255.0
    
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights))
    
    model = build_cnn_model((32, 32, 3))
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=1, validation_data=(X_val, y_val), class_weight=class_weights, callbacks=[early_stopping])
    model.save('/Users/lakshitajawandhiya/Desktop/auto-draw-clone/AdobeGenSolve/data/saved_model.keras')
    model.load_weights('/Users/lakshitajawandhiya/Desktop/auto-draw-clone/AdobeGenSolve/data/saved_model.keras')
    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    print(f"Validation accuracy: {val_accuracy:.4f}")

if __name__ == "__main__":
    main()
