import time
from abc import abstractmethod

import cv2


class VideoCapture:
    def __init__(self, video_path, output_file, **kwargs):
        self.output_file = output_file
        self.video_path = video_path
        self.frame_count = 0
        self.start_time = None
        self.max_frames = None

        self.cap = self.open_video_file()

        (self.frame_width,
         self.frame_height,
         self.fps) = self.get_video_properties()

        self.video_out_writer = self.create_video_output_writer()

    def _validate_file_paths(self):
        if not self.video_path or self.output_file:
            raise AttributeError("Error: Video path and output file must be specified.")

        if not self.video_path.exists():
            raise AttributeError(f"Error: Video file '{self.video_path}' not found.")

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
            if not self._process_frame():
                break
            # self._process_frame()
        self._release_resources()

    # noinspection PyAbstractClass
    @abstractmethod
    def ai_process_frame(self, frame):
        ...

    def _get_ret_and_frame(self):
        ret, frame = self.cap.read()
        return ret, frame

    def _process_frame(self):
        ret, frame = self._get_ret_and_frame()
        if not ret:
            print("No more frames in the video.")
            return False

        # Process the frame
        processed_frame, detections = self.ai_process_frame(frame)
        self._write_and_show_progress(processed_frame)

        # Check if we've reached the maximum number of frames to process
        if self.max_frames is not None and self.frame_count >= self.max_frames:
            print(f"Reached maximum frame limit ({self.max_frames}). Stopping processing.")
            return False
        return True

    def _write_and_show_progress(self, processed_frame):
        # Write the frame to output video
        self.video_out_writer.write(processed_frame)

        # Display progress
        self.frame_count += 1
        if self.frame_count % 100 == 0:
            elapsed_time = time.time() - self.start_time
            fps = self.frame_count / elapsed_time
            print(f"Processed {self.frame_count} frames. FPS: {fps:.2f}")
