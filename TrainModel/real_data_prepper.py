from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from TrainModel import _LabelLoaderMixin


class RealDataPrepper(_LabelLoaderMixin):
    def __init__(self, img_dir:Optional[Path],
                 label_dir:Optional[Path], img_size=640):
        self.detections = []
        super().__init__()
        self.img_dir = img_dir
        self.label_dir = label_dir
        self._check_for_dirs()

        self.img_size = img_size
        self.images = []
        self.labels = []

    def _check_for_dirs(self):
        if self.img_dir is None or self.label_dir is None:
            if type(self) == RealDataPrepper:
                raise ValueError("img_dir and label_dir must be specified.")

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
