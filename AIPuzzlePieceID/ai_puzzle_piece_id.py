
"""
ai_puzzle_piece_id.py

Use AI to identify edge puzzle pieces in a video
"""
from abc import abstractmethod

import cv2
import numpy as np
import argparse
import time
import os
from pathlib import Path

# Try to import tensorflow for model loading and inference
try:
    import tensorflow as tf
except ImportError:
    tf = None

from _version import __version__

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Identify edge puzzle pieces in a video using AI.')
    parser.add_argument('--video', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('--output', type=str, default='output.mp4', help='Path to the output video file.')
    parser.add_argument('--model', type=str, default='model.h5', help='Path to the trained model file.')
    parser.add_argument('--confidence', type=float, default=0.5, help='Minimum confidence threshold for detection.')
    return parser.parse_args()


class VideoCapture:
    def __init__(self, video_path, output_file):
        self.output_file = output_file
        self.video_path = video_path
        self.frame_count = 0
        self.start_time = None
        self.confidence = None
        self.model = None

        if not self.video_path.exists():
            raise AttributeError(f"Error: Video file '{self.video_path}' not found.")

        self.cap = self.open_video_file()

        (self.frame_width,
         self.frame_height,
         self.fps) = self.get_video_properties()

        self.video_out_writer = self.create_video_output_writer()

    def open_video_file(self):
        # Open the video file
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise AttributeError(f"Error: Could not open video file '{self.video_path}'.")
        return cap

    def get_video_properties(self):
        # Get video properties
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return frame_width, frame_height, fps

    def create_video_output_writer(self):
        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_file, fourcc, self.fps,
                              (self.frame_width, self.frame_height))
        return out

    def _release_resources(self):
        # Release resources
        self.cap.release()
        self.video_out_writer.release()
        print(f"Processing complete. Output saved to {self.output_file}")

    def process_frames(self):
        # Process each frame
        self.frame_count = 0
        self.start_time = time.time()

        while self.cap.isOpened():
            self._process_frame()
        self._release_resources()

    # noinspection PyAbstractClass
    @abstractmethod
    def ai_process_frame(self, frame):
        ...

    def _process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Process the frame
        processed_frame, detections = self.ai_process_frame(frame)
        self._write_and_show_progress(processed_frame)

    def _write_and_show_progress(self, processed_frame):
        # Write the frame to output video
        self.video_out_writer.write(processed_frame)

        # Display progress
        self.frame_count += 1
        if self.frame_count % 100 == 0:
            elapsed_time = time.time() - self.start_time
            fps = self.frame_count / elapsed_time
            print(f"Processed {self.frame_count} frames. FPS: {fps:.2f}")


class PuzzlePieceDetector(VideoCapture):
    def __init__(self, model, confidence, video_path, output_file):
        super().__init__(video_path, output_file)
        self.model = model
        if isinstance(self.model, (str, Path)):
            self.model = self.load_model(self.model)
        self.confidence = confidence

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

    def annotate_results(self, frame, results):
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

    def _process_frame(self):
        super()._process_frame()
        # FIXME: Add a break condition to stop processing when desired number of frames is processed
def main():
    """Main function to process video and identify edge puzzle pieces."""
    args = parse_arguments()
    
    # Check if video file exists
    video_path = Path(args.video)

if __name__ == "__main__":
    PPD = PuzzlePieceDetector(model=Path('../Misc_Project_Files/dummy_model.keras'),
                              confidence=0.5,
                              video_path=Path('../Misc_Project_Files/TestVideo2.mp4'),
                              output_file=Path('../Misc_Project_Files/output.mp4'))
    PPD.process_frames()
    #main()