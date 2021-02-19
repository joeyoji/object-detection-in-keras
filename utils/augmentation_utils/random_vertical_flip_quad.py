import cv2
import numpy as np
import random


def random_vertical_flip_quad(
    image,
    quads,
    classes,
    p=0.5
):
    """ Randomly flipped the image vertically. The image format is assumed to be BGR to match Opencv's standard.

    Args:
        - image: numpy array representing the input image.
        - quads: numpy array representing the quadrilaterals.
        - classes: the list of classes associating with each quadrilaterals.
        - p: The probability with which the image is flipped vertically

    Returns:
        - image: The modified image
        - quads: The modified quadrilaterals
        - classes: The unmodified bounding boxes

    Raises:
        - p is smaller than zero
        - p is larger than 1
    """

    assert p >= 0, "p must be larger than or equal to zero"
    assert p <= 1, "p must be less than or equal to 1"

    if (random.random() > p):
        return image, quads, classes

    temp_quads = quads.copy()
    temp_quads[:, :, 1] = image.shape[0] - quads[:, :, 1]
    return np.array(cv2.flip(np.uint8(image), 0), dtype=np.float), temp_quads, classes