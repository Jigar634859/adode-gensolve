from tensorflow.keras.models import load_model

def verify_versions():
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version: {tf.keras.__version__}")

def inspect_model(model_path):
    try:
        model = load_model(model_path)
        model.summary()
        for layer in model.layers:
            print(layer.get_config())
    except Exception as e:
        print(f"Error loading model: {e}")

def resave_model(original_model_path, new_model_path):
    model = load_model(original_model_path)
    model.save(new_model_path)
    print(f"Model re-saved at {new_model_path}")

def load_new_model(new_model_path):
    try:
        model = load_model(new_model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    model_path = '/Users/lakshitajawandhiya/Desktop/auto-draw-clone/AdobeGenSolve/data/saved_model.keras'
    new_model_path = '/Users/lakshitajawandhiya/Desktop/auto-draw-clone/AdobeGenSolve/data/new_saved_model.keras'
    
    print("Verifying TensorFlow and Keras versions...")
    verify_versions()
    
    print("\nInspecting the model...")
    inspect_model(model_path)
    
    print("\nRe-saving the model...")
    resave_model(model_path, new_model_path)
    
    print("\nLoading the newly saved model...")
    load_new_model(new_model_path)
