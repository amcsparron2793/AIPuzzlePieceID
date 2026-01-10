"""
create_test_data.py

Generate synthetic test data for evaluating puzzle piece detection models.
This creates images with simulated puzzle pieces and saves them for testing.
"""

import cv2
import numpy as np
import os
import random
import argparse
from pathlib import Path

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate test data for puzzle piece detection.')
    parser.add_argument('--output-dir', type=str, default='test_data',
                        help='Directory to save generated test images.')
    parser.add_argument('--num-images', type=int, default=10,
                        help='Number of test images to generate.')
    parser.add_argument('--image-size', type=int, default=640,
                        help='Size of the generated images (square).')
    parser.add_argument('--with-annotations', action='store_true',
                        help='Generate annotation files with ground truth.')
    return parser.parse_args()

def create_puzzle_background(img_size, style='plain'):
    """Create a background for the puzzle pieces."""
    if style == 'plain':
        # Create a plain colored background
        bg_color = (
            random.randint(180, 240),
            random.randint(180, 240),
            random.randint(180, 240)
        )
        bg = np.ones((img_size, img_size, 3), dtype=np.uint8) * bg_color
    elif style == 'gradient':
        # Create a gradient background
        bg = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        color1 = np.array([random.randint(180, 240), random.randint(180, 240), random.randint(180, 240)])
        color2 = np.array([random.randint(180, 240), random.randint(180, 240), random.randint(180, 240)])

        for i in range(img_size):
            alpha = i / img_size
            bg[i, :, :] = (1 - alpha) * color1 + alpha * color2
    elif style == 'noise':
        # Create a noisy background
        bg = np.random.randint(180, 240, (img_size, img_size, 3), dtype=np.uint8)
        bg = cv2.GaussianBlur(bg, (21, 21), 0)
    else:
        bg = np.ones((img_size, img_size, 3), dtype=np.uint8) * 200

    return bg

def create_puzzle_piece(shape='rectangle', size_range=(50, 150)):
    """Create a puzzle piece shape."""
    min_size, max_size = size_range
    width = random.randint(min_size, max_size)
    height = random.randint(min_size, max_size)

    # Base shape is a blank canvas
    piece = np.zeros((height, width, 3), dtype=np.uint8)

    if shape == 'rectangle':
        # Simple rectangle
        piece_color = (
            random.randint(30, 150),
            random.randint(30, 150),
            random.randint(30, 150)
        )
        piece[:] = piece_color
    elif shape == 'irregular':
        # Create an irregular shape (simplified)
        piece_color = (
            random.randint(30, 150),
            random.randint(30, 150),
            random.randint(30, 150)
        )
        piece[:] = piece_color

        # Add some random cutouts to simulate puzzle piece shape
        num_cutouts = random.randint(1, 4)
        for _ in range(num_cutouts):
            cx = random.randint(0, width - 10)
            cy = random.randint(0, height - 10)
            cr = random.randint(5, 15)
            cv2.circle(piece, (cx, cy), cr, (0, 0, 0), -1)
    else:
        piece_color = (
            random.randint(30, 150),
            random.randint(30, 150),
            random.randint(30, 150)
        )
        piece[:] = piece_color

    # Add a border to make it look more like a puzzle piece
    cv2.rectangle(piece, (0, 0), (width-1, height-1), (0, 0, 0), 2)

    return piece

def is_edge_piece(index, total_pieces, edge_ratio=0.4):
    """
    Determine if a piece should be an edge piece.
    In a real puzzle, edge pieces are around the perimeter.
    Here we'll just make a percentage of pieces edge pieces.
    """
    # Either make it random or based on index
    return random.random() < edge_ratio

def create_test_image(img_size=640, num_pieces_range=(5, 15)):
    """
    Create a test image with simulated puzzle pieces.

    Args:
        img_size: Size of the square image
        num_pieces_range: Range for the number of pieces to generate

    Returns:
        image: The generated test image
        annotations: List of annotations for each piece [x, y, w, h, is_edge]
    """
    # Create background
    bg_style = random.choice(['plain', 'gradient', 'noise'])
    image = create_puzzle_background(img_size, style=bg_style)

    # Determine number of pieces
    min_pieces, max_pieces = num_pieces_range
    num_pieces = random.randint(min_pieces, max_pieces)

    annotations = []

    for i in range(num_pieces):
        # Create a puzzle piece
        piece_shape = random.choice(['rectangle', 'irregular'])
        piece = create_puzzle_piece(shape=piece_shape)

        # Get piece dimensions
        piece_h, piece_w = piece.shape[:2]

        # Generate a random position
        x = random.randint(0, img_size - piece_w)
        y = random.randint(0, img_size - piece_h)

        # Determine if it's an edge piece
        is_edge = is_edge_piece(i, num_pieces)

        # If it's an edge piece, make it visually distinctive
        if is_edge:
            # Add a distinctive marker for edge pieces (like a colored corner)
            cv2.rectangle(piece, (0, 0), (15, 15), (0, 0, 255), -1)  # Red corner for edge pieces

        # Place the piece on the image
        try:
            image[y:y+piece_h, x:x+piece_w] = piece
        except ValueError:
            # Skip if dimensions don't match (rare case at the border)
            continue

        # Add annotation [x, y, width, height, is_edge]
        # Normalized coordinates for compatibility with models
        x_center = (x + piece_w/2) / img_size
        y_center = (y + piece_h/2) / img_size
        width = piece_w / img_size
        height = piece_h / img_size

        annotations.append([x_center, y_center, width, height, 1.0 if is_edge else 0.0])

    return image, annotations

def save_annotations(annotations, file_path):
    """Save annotations to a text file."""
    with open(file_path, 'w') as f:
        for ann in annotations:
            f.write(' '.join(map(str, ann)) + '\n')

def main():
    args = parse_arguments()

    # Create output directory if it doesn't exist
    images_dir = os.path.join(args.output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    if args.with_annotations:
        annotations_dir = os.path.join(args.output_dir, 'annotations')
        os.makedirs(annotations_dir, exist_ok=True)

    print(f"Generating {args.num_images} test images...")

    for i in range(args.num_images):
        # Generate the test image and annotations
        image, annotations = create_test_image(
            img_size=args.image_size,
            num_pieces_range=(5, 20)
        )

        # Save the image
        image_path = os.path.join(images_dir, f'test_image_{i:04d}.jpg')
        cv2.imwrite(image_path, image)

        if args.with_annotations:
            # Save annotations
            annotation_path = os.path.join(annotations_dir, f'test_image_{i:04d}.txt')
            save_annotations(annotations, annotation_path)

        # Create a visualization with bounding boxes
        visualization = np.array(image.copy(), dtype=np.uint8)
        for ann in annotations:
            x_center, y_center, width, height, is_edge = ann

            # Convert normalized coordinates back to pixel values
            x = int((x_center - width/2) * args.image_size)
            y = int((y_center - height/2) * args.image_size)
            w = int(width * args.image_size)
            h = int(height * args.image_size)

            # Draw bounding box
            color = (0, 255, 0) if is_edge > 0.5 else (255, 0, 0)  # Green for edge, Red for non-edge
            cv2.rectangle(visualization, (x, y), (x + w, y + h), color, 2)

            # Add label
            label = "Edge" if is_edge > 0.5 else "Non-Edge"
            cv2.putText(visualization, label, (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Save the visualization
        vis_dir = os.path.join(args.output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        vis_path = os.path.join(vis_dir, f'vis_test_image_{i:04d}.jpg')
        cv2.imwrite(vis_path, visualization)

        if i % 5 == 0 or i == args.num_images - 1:
            print(f"Generated {i+1}/{args.num_images} images")

    print(f"Test data generation complete. {args.num_images} images created in {args.output_dir}")
    print(f"- Images: {images_dir}")
    if args.with_annotations:
        print(f"- Annotations: {annotations_dir}")
    print(f"- Visualizations: {vis_dir}")

if __name__ == "__main__":
    main()