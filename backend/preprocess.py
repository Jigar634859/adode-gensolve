# backend/preprocess.py
import numpy as np
import json
import matplotlib.pyplot as plt

def strokes_to_image(strokes, size=28):
    image = np.zeros((size, size))
    for stroke in strokes:
        for i in range(len(stroke[0]) - 1):
            x0, y0 = stroke[0][i], stroke[1][i]
            x1, y1 = stroke[0][i + 1], stroke[1][i + 1]
            plt.plot([x0, x1], [y0, y1], 'k-')
    plt.axis('off')
    plt.gca().invert_yaxis()
    return image
