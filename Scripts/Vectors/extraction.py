import os
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from skimage.color import rgb2gray
from sklearn.preprocessing import normalize


def normalize_vector(vec):
    return normalize(np.array(vec).reshape(1, -1))[0].astype("float32").tolist()

def get_object_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def extract_color_histogram(img_path, bins=32):
    image = cv2.imread(img_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = get_object_mask(image)
    h_hist = cv2.calcHist([hsv], [0], mask, [bins], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], mask, [bins], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], mask, [bins], [0, 256]).flatten()
    return normalize_vector(np.concatenate([h_hist, s_hist, v_hist]))

def extract_rgb(img_path, bins=32):
    image = cv2.imread(img_path)
    mask = get_object_mask(image)
    b, g, r = cv2.split(image)
    hist_b = cv2.calcHist([b], [0], mask, [bins], [0, 256]).flatten()
    hist_g = cv2.calcHist([g], [0], mask, [bins], [0, 256]).flatten()
    hist_r = cv2.calcHist([r], [0], mask, [bins], [0, 256]).flatten()
    return normalize_vector(np.concatenate([hist_r, hist_g, hist_b]))

def extract_shape_descriptor(img_path):
    image = cv2.imread(img_path)
    mask = get_object_mask(image)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return normalize_vector(np.zeros(8))
    moments = cv2.moments(mask)
    hu = cv2.HuMoments(moments).flatten()
    hu_log = np.log1p(np.abs(hu[:3]))
    x, y, w, h = cv2.boundingRect(mask)
    area = cv2.countNonZero(mask)
    perimeter = cv2.arcLength(contours[0], True)
    aspect_ratio = float(w) / h if h != 0 else 0
    extent = float(area) / (w * h) if w * h != 0 else 0
    hull = cv2.convexHull(contours[0])
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area != 0 else 0
    features = np.array([area, perimeter, aspect_ratio, extent, solidity, *hu_log])
    return normalize_vector(features)

def extract_hu(img_path):
    return extract_shape_descriptor(img_path)

def extract_texture_descriptor(img_path, P=8, R=1.0):
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = get_object_mask(image)
    lbp = local_binary_pattern(gray, P=P, R=R, method='default')
    masked_lbp = lbp[mask > 0]
    n_bins = int(masked_lbp.max() + 1)
    hist, _ = np.histogram(masked_lbp, bins=n_bins, range=(0, n_bins))
    return normalize_vector(hist)

def extract_hog(img_path):
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fd, _ = hog(gray, orientations=9, pixels_per_cell=(4, 4),
                cells_per_block=(2, 2), visualize=True, block_norm="L2")
    return normalize_vector(fd)
