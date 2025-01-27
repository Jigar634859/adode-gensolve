import numpy as np
from PIL import Image, ImageDraw

def csv_to_strokes(csv_data):
    """
    Convert CSV data to strokes.
    Assumes each line in CSV data is a stroke, and each stroke is a list of (x, y) points.
    """
    strokes = []
    for line in csv_data.split('\n'):
        if line.strip():
            points = line.split(',')
            stroke = []
            for i in range(0, len(points), 2):
                try:
                    x, y = float(points[i]), float(points[i+1])
                    stroke.append([x, y])
                except ValueError:
                    # Handle invalid float conversion
                    continue
            strokes.append(stroke)
    return strokes

def strokes_to_image(strokes):
    """
    Convert strokes to a 28x28 image.
    """
    img = Image.new('L', (28, 28), color=255)  # Create a new grayscale image
    draw = ImageDraw.Draw(img)  # Create a drawing object

    for stroke in strokes:
        if len(stroke) > 1:
            # Convert stroke points to proper coordinates in image space
            scaled_stroke = [(int(x / 10), int(y / 10)) for x, y in stroke]  # Assuming stroke points are scaled
            draw.line(scaled_stroke, fill=0, width=2)

    img_array = np.array(img)
    return img_array

def draw_circle_based_on_input(strokes, image_size=28):
    """
    Draw a circle based on the input strokes' bounding box.
    """
    img = Image.new('L', (image_size, image_size), color=255)  # Create a new grayscale image
    draw = ImageDraw.Draw(img)

    # Flatten strokes into a list of points
    points = [point for stroke in strokes for point in stroke]
    if not points:
        return np.array(img)

    # Calculate the bounding box of the input strokes
    xs, ys = zip(*points)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Determine the circle center and radius based on bounding box
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    radius = max(max_x - min_x, max_y - min_y) / 2

    # Scale the coordinates to fit within the image size
    center_x = (center_x - min_x) / (max_x - min_x) * image_size
    center_y = (center_y - min_y) / (max_y - min_y) * image_size
    radius = radius / max(max_x - min_x, max_y - min_y) * image_size

    # Draw the circle
    draw.ellipse([center_x - radius, center_y - radius, center_x + radius, center_y + radius], outline=0, width=2)

    return np.array(img)

def draw_triangle_based_on_input(strokes, image_size=28):
    """
    Draw a triangle based on the input strokes' bounding box.
    """
    img = Image.new('L', (image_size, image_size), color=255)  # Create a new grayscale image
    draw = ImageDraw.Draw(img)

    # Flatten strokes into a list of points
    points = [point for stroke in strokes for point in stroke]
    if not points:
        return np.array(img)

    # Calculate the bounding box of the input strokes
    xs, ys = zip(*points)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Determine the triangle vertices based on bounding box
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    size = max(max_x - min_x, max_y - min_y)
    height = np.sqrt(3) / 2 * size

    # Scale the coordinates to fit within the image size
    center_x = (center_x - min_x) / (max_x - min_x) * image_size
    center_y = (center_y - min_y) / (max_y - min_y) * image_size
    size = size / max(max_x - min_x, max_y - min_y) * image_size
    height = height / max(max_x - min_x, max_y - min_y) * image_size

    points = [
        (center_x, center_y - height / 2),
        (center_x - size / 2, center_y + height / 2),
        (center_x + size / 2, center_y + height / 2)
    ]

    # Draw the triangle
    draw.polygon(points, outline=0, fill=255)

    return np.array(img)
