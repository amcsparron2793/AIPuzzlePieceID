"""
train_dummy_model.py

Train a dummy model for puzzle piece detection.
"""
from pathlib import Path
from typing import Tuple

import numpy as np
import cv2
import argparse
import random

from TrainModel import CreateAndTrainModel, RealDataPrepper


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


class SyntheticDataGenerator(RealDataPrepper):
    """Generates synthetic data for training the dummy model."""

    def __init__(self, num_samples, img_size=640):
        super().__init__(img_dir=None, label_dir=None, img_size=img_size)
        self.num_samples = num_samples

    # noinspection PyMethodOverriding
    def _normalize_coordinates(self, x: int, w: int, h: int, y_pos: int):
        # Add to detections - normalize coordinates to [0, 1]
        # Format: [x_center, y_center, width, height, confidence]
        x_center = (x + w / 2) / self.img_size
        y_center = (y_pos + h / 2) / self.img_size
        width = w / self.img_size
        height = h / self.img_size
        return x_center, y_center, width, height

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
        is_edge = random.random() > 0.5  # 50% chance of being an edge piece
        confidence = random.uniform(0.7, 0.99) if is_edge else random.uniform(0.3, 0.6)
        return is_edge, confidence

    @staticmethod
    def _draw_rectangle(img, w: int, h: int, x: int, y_pos: int, color: Tuple[int, int, int]):
        # Draw the rectangle
        cv2.rectangle(img, (x, y_pos), (x + w, y_pos + h), color, -1)

        # Add a border to make it look more like a puzzle piece
        cv2.rectangle(img, (x, y_pos), (x + w, y_pos + h), (0, 0, 0), 2)

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

    def _create_pieces(self, num_pieces: int, img):
        for _ in range(num_pieces):
            coordinates, is_edge, confidence = self._create_and_normalize_piece(img)
            # split coordinates into x_center, y_center, width, height
            x_center, y_center, width, height = coordinates
            self.detections.append([x_center, y_center, width, height, confidence])

    def process_entry(self):
        self.detections = []

        img = self._process_data()

        self._post_proc_detections()

        # Add to datasets
        self.images.append(img)
        self.labels.append(self.detections)

    def _process_data(self, img_path:Path = None):
        img, num_pieces = self._create_img()
        self._create_pieces(num_pieces, img)
        return img

    def create_data(self):
        """
        Create synthetic data for training the dummy model.

        This function generates random images with random "puzzle pieces"
        (represented as rectangles) and their corresponding labels.
        """

        for _ in range(self.num_samples):
            self.process_entry()

        self._convert_images_and_labels_to_np_arrays()


class GenCreateAndTrainDummyModel(CreateAndTrainModel):
    def __init__(self, input_shape=(640, 640, 3), t_num_samples=200, v_num_samples=50, **kwargs):
        self.xy_train: tuple = (None, None)
        self.xy_val: tuple = (None, None)
        self.t_generator = None
        self.v_generator = None

        self.input_shape = input_shape
        self.t_num_samples = t_num_samples
        self.v_num_samples = v_num_samples

        self._init_generators()

        self.create_syn_data()

        super().__init__(self.input_shape, self.xy_train, self.xy_val, **kwargs)

        self.model = self.create_model()

    def _init_generators(self):
        self.t_generator = SyntheticDataGenerator(num_samples=self.t_num_samples)
        self.v_generator = SyntheticDataGenerator(num_samples=self.v_num_samples)

    def create_model(self, **kwargs):
        print_model_summary = kwargs.get('print_summary', False)
        self.model = super().create_model()
        if print_model_summary:
            self.print_model_summary()
        return self.model

    def print_model_summary(self):
        if self.model is None:
            raise ValueError("Model not initialized yet.")
        self.model.summary()

    @staticmethod
    def _str_shape(shape_tuple: tuple):
        return ' '.join((str(x.shape) for x in shape_tuple))

    def create_syn_data(self):
        self.t_generator.create_data()
        self.v_generator.create_data()

        self.xy_train = self.t_generator.images, self.t_generator.labels
        self.xy_val = self.v_generator.images, self.v_generator.labels

        print(f"Training data shape: {self._str_shape(self.xy_train)}")
        print(f"Validation data shape: {self._str_shape(self.xy_val)}")


if __name__ == "__main__":
    GCTDummynet = GenCreateAndTrainDummyModel()
    GCTDummynet.train_model()
    GCTDummynet.save_model()