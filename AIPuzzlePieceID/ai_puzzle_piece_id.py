
"""
ai_puzzle_piece_id.py

Use AI to identify edge puzzle pieces in a video
"""
from abc import abstractmethod, ABCMeta

import cv2
import numpy as np
import argparse
from pathlib import Path

from AIPuzzlePieceID import VideoCaptureMixin, ImageCaptureMixin

# Try to import tensorflow for model loading and inference
try:
    import tensorflow as tf
except ImportError:
    tf = None


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Identify edge puzzle pieces in a video using AI.')
    parser.add_argument('--video', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('--output', type=str, default='output.mp4', help='Path to the output video file.')
    parser.add_argument('--model', type=str, default='model.h5', help='Path to the trained model file.')
    parser.add_argument('--confidence', type=float, default=0.5, help='Minimum confidence threshold for detection.')
    return parser.parse_args()


class PuzzlePieceDetectorBase(metaclass=ABCMeta):#(VideoCapture):
    def __init__(self, model, output_file, confidence, **kwargs):
        self.output_file = output_file
        self.model = model
        self.confidence = confidence

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        if isinstance(model, (str, Path)):
            self._model = self.load_model(model)
        else:
            self._model = model

    @staticmethod
    def load_model(model_path):
        return tf.keras.models.load_model(model_path)

    @staticmethod
    def preprocess_frame(frame):
        # 1. Preprocessing the frame (resize, normalize, etc.)
        # Assuming model expects 640x640 input
        input_size = (640, 640)
        resized_frame = cv2.resize(frame, input_size)
        return resized_frame.astype(np.float32) / 255.0

    def run_inference(self, normalized_frame):
        # 2. Running inference with the model
        detections = []
        if self.model is not None:
            # Expand dimensions to create a batch of 1
            input_tensor = np.expand_dims(normalized_frame, axis=0)

            # Run inference
            if hasattr(self.model, 'predict'):
                detections = self.model.predict(input_tensor, verbose=0)
                # Keras model.predict usually returns a batch. We take the first one.
                if len(detections.shape) > 2:
                    detections = detections[0]
            elif callable(self.model):
                # For some model types (like TensorFlow SavedModel)
                detections = self.model(input_tensor)
                if isinstance(detections, tf.Tensor):
                    detections = detections.numpy()
                if len(detections.shape) > 2:
                    detections = detections[0]
        return detections

    def postprocess_frame(self, detections):
        # 3. Post-processing results (filtering by confidence, etc.)
        # Placeholder detections for demonstration if no model is present
        # format: [x, y, w, h, confidence]
        results = []
        for det in detections:
            confidence = det[4]
            if confidence > self.confidence:
                results.append(det)
        return results

    @staticmethod
    def annotate_results(frame, results):
        # 4. Annotating the frame with detections
        annotated_frame = frame.copy()
        h, w, _ = frame.shape
        for res in results:
            x, y, rw, rh, conf = res
            # Scale coordinates back to original frame size
            x1, y1 = int(x * w), int(y * h)
            x2, y2 = int((x + rw) * w), int((y + rh) * h)

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Edge: {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return annotated_frame

    def ai_process_frame(self, frame):
        """
        Process a single frame to identify puzzle pieces.

        Args:
            frame: Input video frame

        Returns:
            Processed frame with annotations
            List of detected edge pieces coordinates
        """
        normalized_frame = self.preprocess_frame(frame)
        detections = self.run_inference(normalized_frame)
        results = self.postprocess_frame(detections)
        annotated_frame = self.annotate_results(frame, results)

        return annotated_frame, results

    @abstractmethod
    def _process_frame(self):
        ...


class PuzzlePieceDetectorVideo(PuzzlePieceDetectorBase, VideoCaptureMixin):
    def __init__(self, model, output_file, confidence, video_path, **kwargs):
        # Call both parent class initializers
        VideoCaptureMixin.__init__(self, video_path, output_file, **kwargs)
        PuzzlePieceDetectorBase.__init__(self, model, output_file, confidence, **kwargs)

    def _process_frame(self):
        return VideoCaptureMixin._process_frame(self)


class PuzzlePieceDetectorImage(PuzzlePieceDetectorBase, ImageCaptureMixin):
    def __init__(self, model, output_file, confidence, image_dir, **kwargs):
        # FIXME: make sure that all of these attrs are needed
        ImageCaptureMixin.__init__(self, image_dir=image_dir, output_dir='', output_file=output_file, **kwargs)
        PuzzlePieceDetectorBase.__init__(self, model, output_file, confidence, **kwargs)

    def _process_frame(self):
        return ImageCaptureMixin._process_frame(self)


def main():
    """Main function to process video and identify edge puzzle pieces."""
    args = parse_arguments()
    
    # Check if video file exists
    video_path = Path(args.video)

if __name__ == "__main__":
    # PPD = PuzzlePieceDetectorVideo(model=Path('../Misc_Project_Files/dummy_model.keras'),
    #                           confidence=0.5,
    #                           video_path=Path('../Misc_Project_Files/TestVideo2.mp4'),
    #                           output_file=Path('../Misc_Project_Files/output.mp4'),
    #                           max_frames=500)
    PPD = PuzzlePieceDetectorImage(model=Path('../Misc_Project_Files/dummy_model.keras'), confidence=0.5,
                                   image_dir=Path('../Misc_Project_Files'),
                                   output_file='')
    PPD.process_frames()
    #main()