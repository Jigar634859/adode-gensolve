import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

model_path = '/Users/lakshitajawandhiya/Desktop/auto-draw-clone/AdobeGenSolve/data/saved_model.keras'
new_model_path = '/Users/lakshitajawandhiya/Desktop/auto-draw-clone/AdobeGenSolve/data/new_saved_model.keras'

def create_model():
    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))  # Adjust the input shape as per your model
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))  # Adjust the number of classes as per your model
    return model

try:
    print("Reconstructing the model architecture...")
    model = create_model()
    model.summary()

    print("Loading weights...")
    model.load_weights(model_path)
    print("Weights loaded successfully!")

    print("Saving the reconstructed model...")
    model.save(new_model_path)
    print(f"Model re-saved at {new_model_path}")

    print("Loading the newly saved model...")
    new_model = load_model(new_model_path)
    print("New model loaded successfully!")

except Exception as e:
    print(f"Error: {e}")
