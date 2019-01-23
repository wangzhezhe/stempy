from stempy import _image
import numpy as np
from collections import namedtuple

def create_stem_image(blocks, width, height,  inner_radius, outer_radius):
    img =  _image.create_stem_image([b._block for b in blocks],
                                    width, height,  inner_radius, outer_radius)

    image = namedtuple('STEMImage', ['bright', 'dark'])
    image.bright = np.array(img.bright, copy = False)
    image.dark = np.array(img.dark, copy = False)

    return image

def create_dark_field_reference(blocks, width, height, number_of_samples=20, strip_width=100):
    dark = _image.create_dark_field_reference([b._block for b in blocks],
                                              width, height, number_of_samples, strip_width)

    dark_reference = namedtuple('DarkFieldReference', ['frame', 'mean', 'variance'])
    dark_reference.frame = np.array(dark, copy = False)
    dark_reference.mean = dark.mean
    dark_reference.variance = dark.variance

    return dark_reference
