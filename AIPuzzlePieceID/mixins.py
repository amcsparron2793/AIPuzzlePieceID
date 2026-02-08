import time
from abc import abstractmethod, ABCMeta
import cv2
from pathlib import Path


class _BaseMixin:
    RELEASE_RESOURCES_STR = "Processing complete. Output saved to {}"

    def __init__(self, in_file, output_file, **kwargs):
        self.in_file = in_file
        self.output_file = output_file
        self.output_dir = Path(output_file).parent
        self.frame_count = 0
        self.start_time = None
        self.max_frames = kwargs.get('max_frames', None)

    def isOpened(self):
        return

    def process_frames(self):
        # Process each frame
        self.frame_count = 0
        self.start_time = time.time()

        while self.isOpened():
            if not self._process_frame():
                break
            # self._process_frame()
        self._release_resources()

    def _process_frame(self):
        return

    def _validate_file_paths(self):
        if not self.in_file or self.output_file:
            raise AttributeError("Error: in_file path and output file must be specified.")

        if not self.in_file.exists():
            raise AttributeError(f"Error: file '{self.in_file}' not found.")

    def _release_resources(self):
        """Cleanup after processing all images."""
        print(self.__class__.RELEASE_RESOURCES_STR.format(self.output_file))

    def _display_progress(self):
        # Display progress
        self.frame_count += 1
        if self.frame_count % 100 == 0:
            elapsed_time = time.time() - self.start_time
            fps = self.frame_count / elapsed_time
            print(f"Processed {self.frame_count} frames. FPS: {fps:.2f}")

    def write(self, frame):
        return True

    def _write_and_show_progress(self, processed_frame):
        self.write(processed_frame)
        self._display_progress()

    def _set_frame_dimensions(self):
        ...


class VideoCaptureMixin(_BaseMixin, metaclass=ABCMeta):
    def __init__(self, video_path, output_file, **kwargs):
        super().__init__(in_file=video_path,
                         output_file=output_file, **kwargs)
        self.video_path = self.in_file
        self.cap = self.open_video_file()

        (self.frame_width,
         self.frame_height,
         self.fps) = self.get_video_properties()

        self.video_out_writer = self.create_video_output_writer()

    def isOpened(self):
        return self.cap.isOpened()

    def open_video_file(self):
        # Open the video file
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise AttributeError(f"Error: Could not open video file '{self.video_path}'.")
        return cap

    def _set_frame_dimensions(self):
        # Get video properties
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return frame_width, frame_height

    def get_video_properties(self):
        frame_width, frame_height = self._set_frame_dimensions()
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return frame_width, frame_height, fps

    # noinspection PyUnresolvedReferences
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
        super()._release_resources()

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

    def write(self, frame):
        # Write the frame to output video
        self.video_out_writer.write(frame)
        return True


class ImageCaptureMixin(_BaseMixin, metaclass=ABCMeta):
    """
    A class similar to VideoCapture but processes image files instead of videos.
    Simulates video-like sequential processing by iterating through a directory of images.
    """

    def __init__(self, image_dir, output_dir, output_file, image_pattern="*.jpg", **kwargs):
        """
        Initialize the ImageCapture.

        Args:
            image_dir (str or Path): Path to the directory containing input images
            output_dir (str or Path): Path to the directory for output images
            image_pattern (str): Glob pattern to filter image files (default: "*.jpg")
            **kwargs: Additional keyword arguments
        """
        in_file = None
        super().__init__(in_file, output_file, **kwargs)

        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.image_pattern = image_pattern

        self.total_images = len(self.image_files)
        self.current_idx = 0

        # Make sure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        self._set_frame_dimensions()

    def _set_frame_dimensions(self):
        # Get frame dimensions from first image (if available)
        if self.total_images > 0:
            sample_img = cv2.imread(str(self.image_files[0]))
            if sample_img is not None:
                self.frame_height, self.frame_width = sample_img.shape[:2]
            else:
                self.frame_width, self.frame_height = 0, 0
        else:
            self.frame_width, self.frame_height = 0, 0
        return self.frame_width, self.frame_height

    @property
    def image_files(self):
        return sorted(self.image_dir.glob(self.image_pattern))

    def _validate_file_paths(self):
        """Validate that the input directory exists and contains images."""
        if not self.image_dir.exists() or not self.image_dir.is_dir():
            raise AttributeError(f"Error: Image directory '{self.image_dir}' not found or is not a directory.")

        if self.total_images == 0:
            raise AttributeError(
                f"Error: No images matching pattern '{self.image_pattern}' found in '{self.image_dir}'.")

    def isOpened(self):
        """Mimics cv2.VideoCapture.isOpened()."""
        return self.current_idx < self.total_images

    def _get_ret_and_frame(self):
        """Gets the next image in the sequence, similar to cv2.VideoCapture.read()."""
        if self.current_idx >= self.total_images:
            return False, None

        image_path = self.image_files[self.current_idx]
        frame = cv2.imread(str(image_path))

        if frame is None:
            print(f"Warning: Could not read image '{image_path}'")
            self.current_idx += 1
            return self._get_ret_and_frame()

        self.current_idx += 1
        return True, frame

    def write(self, frame):
        """Saves the processed frame as an image file."""
        # Generate output filename based on the current frame index
        output_path = self.output_dir / f"frame_{self.frame_count:04d}.jpg"
        cv2.imwrite(str(output_path), frame)
        return True

    @abstractmethod
    def ai_process_frame(self, frame):
        """
        Abstract method to be implemented by derived classes.

        Args:
            frame: Input image

        Returns:
            Tuple containing:
                - Processed image
                - Detected objects or other results
        """
        ...

    def _process_frame(self):
        """Process a single image frame."""
        ret, frame = self._get_ret_and_frame()
        if not ret:
            print("No more images to process.")
            return False

        # Process the frame
        processed_frame, detections = self.ai_process_frame(frame)
        self._write_and_show_progress(processed_frame)

        # Check if we've reached the maximum number of frames to process
        if self.max_frames is not None and self.frame_count >= self.max_frames:
            print(f"Reached maximum frame limit ({self.max_frames}). Stopping processing.")
            return False
        return True

    def _display_progress(self):
        # Display progress
        self.frame_count += 1
        if self.frame_count % 10 == 0:  # Show progress every 10 images
            elapsed_time = time.time() - self.start_time
            imgs_per_sec = self.frame_count / elapsed_time
            print(f"Processed {self.frame_count}/{self.total_images} images. "
                  f"Rate: {imgs_per_sec:.2f} imgs/sec")
