import copy
import numpy as np
from PIL import Image

def numpy_to_pil(image_):
    image = copy.deepcopy(image_)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    image *= 255
    image = image.astype('uint8')

    im_obj = Image.fromarray(image)
    return im_obj
