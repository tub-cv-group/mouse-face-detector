"""
    Part of Mouse Face Detector

    Author: Manuel Wöllhaf
    Contact: n.andresen@tu-berlin.de, andresen.niek@gmail.com
    Affiliation: TU Berlin, Computer Vision & Remote Sensing
    License: MIT

    Part of the publication:
    "Automatic pain face analysis in mice: Applied to a varied dataset with non-standardized conditions"
    DOI: not yet published

    Created: 2018-09-01
    Last Modified: 2026-01-30

    bbox = r_min, r_max, c_min, c_max
"""

from math import floor, ceil
import numpy as np


def hw(bbox):
    """ Height and width of bounding box.

    Args:
      bbox: tuple (r_min, r_max, c_min, c_max)

    Returns:
      hw: tuple (h, w)
    """
    return bbox[1] - bbox[0], bbox[3] - bbox[2]


def center(bbox):
    """ Center of bounding box.

    Args:
      bbox: tuple (r_min, r_max, c_min, c_max)

    Returns:
      center: tuple (r, c)
    """
    h, w = hw(bbox)
    return int(bbox[0] + h//2), int(bbox[2] + w//2)


def from_coords(coords):
    """ Bounding box from coords.

    Args:
      coords: numpy array (n, 2)
        Coordinates.

    Returns:
      bbox: tuple (r_min, r_max, c_min, c_max)
    """
    return floor(coords[:, 0].min()), ceil(coords[:, 0].max()), floor(coords[:, 1].min()), ceil(coords[:, 1].max())


def from_mask(mask, label=None):
    """ Bounding box from pixel mask.

    Args:
      mask: numpy array
        Binary mask or label image.
      label: int (optional)
        Label if mask is label image.

    Returns:
      bbox: tuple (r_min, r_max, c_min, c_max)
    """
    mask = mask if label is None else mask == label
    coords = np.where(mask)
    return coords[0].min(), coords[1].min(), coords[0].max(), coords[1].max()


def iou(bboxa, bboxb):
    """ Intersection over union of bounding boxes bboxa and bboxb.

    Args:
      bboxa: tuple (r_min, r_max, c_min, c_max)
      bboxb: tuple (r_min, r_max, c_min, c_max)

    Returns:
      iou: float
        area(bboxa n bboxb)/area(bboxa u bboxb)
    """
    r_min_a, r_max_a, c_min_a, c_max_a = bboxa
    r_min_b, r_max_b, c_min_b, c_max_b = bboxb
    cs, rs = max(c_min_a, c_min_b), max(r_min_a, r_min_b)
    ce, ye = min(c_max_a, c_max_b), min(r_max_a, r_max_b)
    areai = (ce-cs)*(ye-rs) if ce > cs and ye > rs else 0.0
    ha, wa = hw(bboxa)
    hb, wb = hw(bboxb)
    areau = ha*wa + hb*wb - areai
    return float(areai)/areau


def has_intersection(bboxa, bboxb):
    """ Returns True if bboxa has an intersection with bboxb.

    Args:
      bboxa: tuple (r_min, r_max, c_min, c_max)
      bboxb: tuple (r_min, r_max, c_min, c_max)

    Returns:
      iou: bool
    """
    r_min_a, r_max_a, c_min_a, c_max_a = bboxa
    r_min_b, r_max_b, c_min_b, c_max_b = bboxb
    cs, rs = max(c_min_a, c_min_b), max(r_min_a, r_min_b)
    ce, re = min(c_max_a, c_max_b), min(r_max_a, r_max_b)
    return ce > cs and re > rs


def make_square(bbox):
    """ Returns the minimum square bounding box containing bbox.

    Args:
      bbox: tuple (r_min, r_max, c_min, c_max)

    Returns:
      bbox: tuple (r_min, r_max, c_min, c_max)
    """
    r_min, r_max, c_min, c_max = bbox
    dr = r_max - r_min
    dc = c_max - c_min
    ds = max(dr, dc)
    r_min = r_min - (ds - dr)//2
    c_min = c_min - (ds - dc)//2
    r_max = r_min + ds
    c_max = c_min + ds
    return r_min, r_max, c_min, c_max


def maximize(bbox, shape):
    """ Maximize bounding box while keeping center and ratio.

    Args:
      bbox: tuple (r_min, r_max, c_min, c_max)
      shape: tuple of ints
        Max shape.

    Returns:
      bbox: tuple (r_min, r_max, c_min, c_max)

    UNTESTED

    """
    print("UNTESTED CODE")

    h, w = hw(bbox)/2

    min_scale = min([
        bbox[0]/h,
        (shape[0] - bbox[1])/h,
        bbox[2]/w,
        (shape[1] - bbox[2])/w
    ])

    dr, dc = min_scale*h, min_scale*w
    return bbox[0] - dr, bbox[1] + dr, bbox[2] - dc, bbox[3] + dc


def scale(bbox, scale):
    """ Scale bounding box.

    Args:
      bbox: tuple (r_min, r_max, c_min, c_max)
      scale: tuple of floats (scale_row, scale_col)
        Scale factor.

    Returns:
      bbox: tuple (r_min, r_max, c_min, c_max)
    """
    h, w = hw(bbox)
    dh = h*scale[0] - h
    dw = w*scale[1] - w
    r_min = int(bbox[0] - dh//2)
    r_max = int(bbox[1] + round(dh/2))
    c_min = int(bbox[2] - dw//2)
    c_max = int(bbox[3] + round(dw/2))
    return r_min, r_max, c_min, c_max


def fit(bbox, shape=(None, None)):
    """ Crop bounding box to fit image shape.

    Args:
      bbox: tuple (r_min, r_max, c_min, c_max)
      shape: tuple of ints
        Max shape.

    Returns:
      bbox: tuple (r_min, r_max, c_min, c_max)
    """
    r_min = max(bbox[0], 0)
    r_max = bbox[1] if shape[0] is None else min(bbox[1], shape[0])
    c_min = max(bbox[2], 0)
    c_max = bbox[3] if shape[1] is None else min(bbox[3], shape[1])
    return r_min, r_max, c_min, c_max


def fit_with_moving(bbox, shape=(None, None)):
    """ Move bounding box to fit image shape. Crop if necessary.

    Args:
      bbox: tuple (r_min, r_max, c_min, c_max)
      shape: tuple of ints
        Max shape.

    Returns:
      bbox: tuple (r_min, r_max, c_min, c_max)
    """
    r_shift = min(shape[0]-bbox[1], 0) - min(bbox[0], 0)
    c_shift = min(shape[1]-bbox[2], 0) - min(bbox[1], 0)
    bbox = bbox[0] + r_shift, bbox[1] + r_shift, bbox[2] + c_shift, bbox[3] + c_shift
    return fit(bbox, shape)


def minimum_containing_box(bboxes):
    """ Returns the minimum bounding box containing all boxes in bboxes.

    Args:
      bbox: list of tuples [(r_min, r_max, c_min, c_max), ...]

    Returns:
      bbox: tuple (r_min, r_max, c_min, c_max)
    """
    r_min, r_max, c_min, c_max = [], [], [], []
    for bbox in bboxes:
        r_min.append(bbox[0])
        r_max.append(bbox[1])
        c_min.append(bbox[2])
        c_max.append(bbox[3])
    return min(r_min), max(r_max), min(c_min), max(c_max)


def crop_image(img, bbox):
    """ Returns the croped image using bbox.
    Image smaller than bbox if bbox not fully within image shape.

    Args:
      img: numpy array
      bbox: tuple (r_min, r_max, c_min, c_max)

    Returns:
      img: numpy array
    """
    shape = img.shape
    r_min, r_max, c_min, c_max = bbox
    r_min, r_max, c_min, c_max = max(r_min, 0), min(r_max, shape[0]), max(c_min, 0), min(c_max, shape[1])
    return img[r_min:r_max, c_min:c_max, :]


def crop_image_with_padding(img, bbox):
    """ Returns the croped image using bbox and pads image symmetrically if smaller than bbox.

    Args:
      img: numpy array
      bbox: tuple (r_min, r_max, c_min, c_max)

    Returns:
      img: numpy array
    """
    shape = img.shape
    r_min, r_max, c_min, c_max = bbox
    r_min, r_max, c_min, c_max = max(r_min, 0), min(r_max, shape[0]), max(c_min, 0), min(c_max, shape[1])
    img = img[r_min:r_max, c_min:c_max, :]
    pad = ((r_min - bbox[0], bbox[1] - r_max), (c_min - bbox[2], bbox[3] - c_max), (0, 0))
    img = np.pad(img, pad, mode='constant')
    return img
