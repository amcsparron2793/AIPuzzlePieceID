"""
train_dummy_model.py

Train a dummy model for puzzle piece detection.
"""

import os
from typing import Tuple

import numpy as np
from tensorflow.keras import layers, models
import cv2
import argparse
import random


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a dummy model for puzzle piece detection.')
    parser.add_argument('--data', type=str, default='training_data',
                        help='Path to the training data directory.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs for training.')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for training.')
    parser.add_argument('--output', type=str, default='dummy_model.keras',
                        help='Output path for the trained model.')
    return parser.parse_args()


class SyntheticDataGenerator:
    """Generates synthetic data for training the dummy model."""
    def __init__(self, num_samples=100, img_size=640):
        self.num_samples = num_samples
        self.img_size = img_size
        self.images = []
        self.labels = []  # detections

    def _create_img(self):
        # Create a blank image
        shp = (self.img_size, self.img_size, 3)
        img = np.zeros(shp, dtype=np.uint8)

        # Add a random background color
        bg_color = (
            random.randint(0, 100),
            random.randint(0, 100),
            random.randint(0, 100)
        )
        img[:] = bg_color

        # Create 2-10 random "puzzle pieces" (rectangles)
        num_pieces = random.randint(2, 10)
        return img, num_pieces

    def _get_random_size_and_pos_for_piece(self):
        # Random size for the piece
        w = random.randint(50, 150)
        h = random.randint(50, 150)

        # Random position
        x = random.randint(0, self.img_size - w - 1)
        y_pos = random.randint(0, self.img_size - h - 1)
        return w, h, x, y_pos

    @staticmethod
    def _get_edge_and_confidence():
        # Determine if it's an edge piece (just for simulation)
        is_edge = random.random() > 0.5  # 50% chance of being edge piece
        confidence = random.uniform(0.7, 0.99) if is_edge else random.uniform(0.3, 0.6)
        return is_edge, confidence

    @staticmethod
    def _draw_rectangle(img, w: int, h: int, x: int, y_pos: int, color: Tuple[int, int, int]):
        # Draw the rectangle
        cv2.rectangle(img, (x, y_pos), (x + w, y_pos + h), color, -1)

        # Add a border to make it look more like a puzzle piece
        cv2.rectangle(img, (x, y_pos), (x + w, y_pos + h), (0, 0, 0), 2)

    def _normalize_coordinates(self, x: int, w: int, h: int, y_pos: int):
        # Add to detections - normalize coordinates to [0, 1]
        # Format: [x_center, y_center, width, height, confidence]
        x_center = (x + w / 2) / self.img_size
        y_center = (y_pos + h / 2) / self.img_size
        width = w / self.img_size
        height = h / self.img_size
        return x_center, y_center, width, height

    def _create_and_normalize_piece(self, img):
        w, h, x, y_pos = self._get_random_size_and_pos_for_piece()
        is_edge, confidence = self._get_edge_and_confidence()

        # Choose a color - brighter colors for pieces
        color = (
            random.randint(150, 255),
            random.randint(150, 255),
            random.randint(150, 255)
        )
        self._draw_rectangle(img, w, h, x, y_pos, color)

        #x_center, y_center, width, height = self._normalize_coordinates(x, w, h, y_pos)
        return self._normalize_coordinates(x, w, h, y_pos), is_edge, confidence

    def create_synthetic_data(self, num_samples=100, img_size=640):
        """
        Create synthetic data for training the dummy model.

        This function generates random images with random "puzzle pieces"
        (represented as rectangles) and their corresponding labels.
        """
        X = []  # Images
        y = []  # Labels (detections)

        for _ in range(num_samples):
            detections = []
            img, num_pieces = self._create_img()

            for _ in range(num_pieces):
                coordinates, is_edge, confidence = self._create_and_normalize_piece(img)
                # split coordinates into x_center, y_center, width, height
                x_center, y_center, width, height = coordinates
                detections.append([x_center, y_center, width, height, confidence])

            # Ensure we have exactly 10 detections (padding with zeros if needed)
            while len(detections) < 10:
                detections.append([0, 0, 0, 0, 0])  # Zero-padding for unused detections

            # Take only first 10 if we have more
            detections = detections[:10]

            # Add to datasets
            self.images.append(img)
            self.labels.append(detections)

        # Convert to numpy arrays
        self.images = np.array(self.images, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        self.labels = np.array(self.labels, dtype=np.float32)


def create_model(input_shape=(640, 640, 3)):
    """Create the model architecture."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.GlobalAveragePooling2D(),
        # Output layer: 10 detections with 5 values each
        layers.Dense(10 * 5, activation='sigmoid'),
        layers.Reshape((10, 5))
    ])

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['accuracy']
    )

    return model


def load_real_data(data_path, img_size=640):
    """Load real training data from directory."""
    images_dir = os.path.join(data_path, 'images')
    labels_dir = os.path.join(data_path, 'labels')

    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])

    X = []
    y = []

    for img_file in image_files:
        # Load and preprocess image
        img_path = os.path.join(images_dir, img_file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_size, img_size))
        img = img / 255.0  # Normalize

        # Load labels if they exist
        base_name = os.path.splitext(img_file)[0]
        label_path = os.path.join(labels_dir, f"{base_name}.txt")

        detections = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    values = list(map(float, line.strip().split()))
                    # Assuming format: [x_center, y_center, width, height, confidence]
                    detections.append(values)

        # Ensure we have exactly 10 detections
        while len(detections) < 10:
            detections.append([0, 0, 0, 0, 0])
        detections = detections[:10]

        X.append(img)
        y.append(detections)

    return np.array(X), np.array(y)


def main():
    """Main function to train the model."""
    args = parse_arguments()

    print("Creating synthetic training data...")
    X_train, y_train = create_synthetic_data(num_samples=200)
    X_val, y_val = create_synthetic_data(num_samples=50)

    print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    print(f"Validation data shape: {X_val.shape}, {y_val.shape}")

    # Create and compile the model
    print("Creating model...")
    model = create_model()
    model.summary()

    # Train the model
    print(f"Training model for {args.epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1
    )

    # Save the model
    print(f"Saving model to {args.output}...")
    model.save(args.output)
    print("Training complete!")



if __name__ == "__main__":
    main()