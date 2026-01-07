import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def create_puzzle_model(output_path='../Misc_Project_Files/dummy_model.keras'):
    """
    Creates a simple dummy model for puzzle piece detection.
    Expected input: (640, 640, 3)
    Expected output: (N, 5) where N is number of detections and columns are [x, y, w, h, confidence]
    """
    input_shape = (640, 640, 3)
    
    # Define a simple CNN
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.GlobalAveragePooling2D(),
        # For simplicity, let's say it always outputs 10 "detections"
        # Each detection has 5 values: x, y, w, h, confidence
        layers.Dense(10 * 5, activation='sigmoid'),
        layers.Reshape((10, 5))
    ])

    model.compile(optimizer='adam', loss='mse')
    
    print(f"Saving dummy model to {output_path}...")
    model.save(output_path)
    print("Model saved successfully.")

if __name__ == "__main__":
    create_puzzle_model()
