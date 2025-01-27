import numpy as np
from PIL import Image, ImageDraw  # Import ImageDraw

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
                x, y = int(points[i]), int(points[i+1])
                stroke.append([x, y])
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
