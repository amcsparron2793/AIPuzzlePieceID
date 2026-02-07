"""
train_dummy_model.py

Train a dummy model for puzzle piece detection.
"""
import json
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from smart_open.utils import TextIOWrapper
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

# def load_real_data(data_path, img_size=640):
#     """Load real training data from directory."""
#     images_dir = os.path.join(data_path, 'images')
#     labels_dir = os.path.join(data_path, 'labels')
#
#     image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
#
#     X = []
#     y = []
#
#     for img_file in image_files:
#         # TODO: DONE
#         # Load and preprocess image
#         # img_path = os.path.join(images_dir, img_file)
#         # img = cv2.imread(img_path)
#         # img = cv2.resize(img, (img_size, img_size))
#         # img = img / 255.0  # Normalize
#
#         # Load labels if they exist
#         base_name = os.path.splitext(img_file)[0]
#         label_path = os.path.join(labels_dir, f"{base_name}.txt")
#
#         detections = []
#         if os.path.exists(label_path):
#             with open(label_path, 'r') as f:
#                 lines = f.readlines()
#                 for line in lines:
#                     values = list(map(float, line.strip().split()))
#                     # Assuming format: [x_center, y_center, width, height, confidence]
#                     detections.append(values)
#
#         # Ensure we have exactly 10 detections
#         while len(detections) < 10:
#             detections.append([0, 0, 0, 0, 0])
#         detections = detections[:10]
#
#         X.append(img)
#         y.append(detections)
#
#     return np.array(X), np.array(y)


class RealDataPrepper:
    def __init__(self, img_dir:Optional[Path], label_dir:Optional[Path], img_size=640):
        self.img_dir = img_dir
        self.label_dir = label_dir
        if self.img_dir is None or self.label_dir is None:
            if self.__class__.__name__ == 'RealDataPrepper':
                raise ValueError("img_dir and label_dir must be specified.")
        self.img_size = img_size
        self.images = []
        self.labels = []
        self.detections = []

    def _normalize_coordinates(self, img_path):
        # Add to detections - normalize coordinates to [0, 1]
        # Format: [x_center, y_center, width, height, confidence]
        # Load and preprocess image
        #img_path = os.path.join(images_dir, img_file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img / 255.0  # Normalize
        return img

    def _post_proc_detections(self):
        # Ensure we have exactly 10 detections (padding with zeros if needed)
        while len(self.detections) < 10:
            self.detections.append([0, 0, 0, 0, 0])  # Zero-padding for unused detections

        # Take only first 10 if we have more
        self.detections = self.detections[:10]

    def _convert_images_and_labels_to_np_arrays(self):
        # Convert to numpy arrays
        self.images = np.array(self.images, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        self.labels = np.array(self.labels, dtype=np.float32)

    def _process_data(self, img_path:Path = None):
        self._normalize_coordinates(img_path)

    def create_data(self):
        for _ in self.images:
            self.process_entry()
        self._convert_images_and_labels_to_np_arrays()

    def process_entry(self):
        self.detections = []

        img = self._process_data()
        self.load_labels()
        self._post_proc_detections()

        # Add to datasets
        self.images.append(img)
        self.labels.append(self.detections)

    def _load_from_plaintext(self, f:TextIOWrapper):
        lines = f.readlines()
        for line in lines:
            values = list(map(float, line.strip().split()))
            # Assuming format: [x_center, y_center, width, height, confidence]
            self.detections.append(values)

    def _load_from_json(self, f:TextIOWrapper):
        item = json.load(f)
        # Each image can have multiple annotation sets, but typically just one
        for annotation_set in item["annotations"]:
            # Each annotation set has multiple results (individual bounding boxes)
            for result in annotation_set["result"]:
                # Extract bounding box information
                value = result["value"]
                x = value["x"] / 100.0  # Convert from percentage to [0,1]
                y = value["y"] / 100.0
                width = value["width"] / 100.0
                height = value["height"] / 100.0

                # Determine if it's an edge piece
                is_edge = 1.0 if "Edge" in value["rectanglelabels"][0] else 0.0
                # Assuming format: [x_center, y_center, width, height, confidence]
                values = [x, y, width, height, is_edge]
                self.detections.append(values)

    def load_labels(self, img_file:Path = None, **kwargs):
        # Load labels if they exist
        #base_name = os.path.splitext(img_file)[0]
        base_name = kwargs.get('base_name', img_file.stem)
        file_extension = kwargs.get('file_extension', '.txt')
        label_path = Path(self.label_dir,
                          f"{base_name}{file_extension}")

        if label_path.exists():
            with open(label_path, 'r') as f:
                if file_extension == '.json':
                    self._load_from_json(f)
                elif file_extension == '.txt':
                    self._load_from_plaintext(f)



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


class CreateAndTrainModel:
    def __init__(self, input_shape, train_xy:tuple, val_xy:tuple, **kwargs):
        self.x_images_train, self.y_labels_train = train_xy
        self.x_images_val, self.y_labels_val = val_xy
        self.batch_size = kwargs.get('batch_size', 4)

        self.input_shape = input_shape
        self.activation_2D = 'relu'
        self.activation_dense = 'sigmoid'
        self.padding_2D = 'same'
        self.kernel_size_2D = (3, 3)
        self.max_pool_2D = (2, 2)
        self.filters = 16

        self.model = None
        self.epochs = kwargs.get('epochs', 10)
        self.model_output_path = kwargs.get('output', 'dummy_model.keras')

    def create_model(self):
        """Create the model architecture."""
        activation_and_padding_2d = {'kernel_size':self.kernel_size_2D,
                                     'activation': self.activation_2D,
                                     'padding': self.padding_2D}
        self.model = models.Sequential([
            layers.Input(shape=self.input_shape),
            layers.Conv2D(self.filters, **activation_and_padding_2d),
            layers.MaxPooling2D(self.max_pool_2D),
            layers.Conv2D(self.filters * 2, **activation_and_padding_2d),
            layers.MaxPooling2D(self.max_pool_2D),
            layers.Conv2D(self.filters * 4, **activation_and_padding_2d),
            layers.GlobalAveragePooling2D(),
            # Output layer: 10 detections with 5 values each
            layers.Dense(10 * 5, activation=self.activation_dense),
            layers.Reshape((10, 5))
        ])

        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['accuracy']
        )

        return self.model

    def train_model(self):
        # Train the model
        print(f"Training model for {self.epochs} epochs...")
        history = self.model.fit(
            self.x_images_train, self.y_labels_train,
            validation_data=(self.x_images_val, self.y_labels_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=1
        )
        return history

    def save_model(self, **kwargs):
        save_path = kwargs.get('save_path', self.model_output_path)
        # Save the model
        print(f"Saving model to {save_path}...")
        self.model.save(save_path)
        print("Training complete!")


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
        ' '.join((str(x.shape) for x in shape_tuple))

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