import numpy as np
import cv2
import pywt
import os
from os.path import splitext

def is_supported(filename):
  ext = [".bmp", ".dib", ".jpeg", ".jpg", ".jp2", ".png", ".webp", ".pbm", 
  ".pgm", ".ppm", ".pxm", ".pnm", ".tiff", ".tif", ".exr", ".hdr", ".pic"]
  return splitext(filename)[1].lower() in ext

def load_image(filename):
  return cv2.imread(filename)

def save_image(image, filename):
  cv2.imwrite(filename, image)

def resize_image(image, dim):
  return cv2.resize(image, dim, interpolation = cv2.INTER_LINEAR)

# Default colour soace
def get_rgb_channels(image):
  b, g, r = cv2.split(image)
  return [r, g, b]

# Component colour space
def get_cmp_channels(image):
  cmp_max = 255
  r, g, b = get_rgb_channels(image)
  i = (r + g + b) / 3 # intensity
  c1 = (r + (cmp_max - b)) / 2 # contrast 1
  c2 = (r + 2*(cmp_max - g) + b) / 4 # contrast 2
  return [i, c1, c2]

# Opponent colour space
def get_opp_channels(image):
  r, g, b = get_rgb_channels(image)
  rg = r - 2*g + b
  by = -r - g + 2*b
  wb = r + g + b
  return [rg, by, wb]

def get_dwt(channels, level=5):
  coeffs = channels
  for c in range(len(channels)):
    for l in range(level):
      coeff = pywt.dwt2(coeffs[c] if l == 0 else coeffs[c][0], wavelet="bior1.3")
      coeffs[c] = np.array([coeff[0], coeff[1][0], coeff[1][1], coeff[1][2]])
  return coeffs

def get_feature_vector(cmp_dwt, rgb_dwt):
  vector = {
      'w_c1': cmp_dwt[0],
      'w_c2': cmp_dwt[1],
      'w_c3': cmp_dwt[2],
      'w_c4': rgb_dwt[0],
      'w_c5': rgb_dwt[1],
      'w_c6': rgb_dwt[2],
      's_c1': np.std(cmp_dwt[0][0]),
      's_c2': np.std(cmp_dwt[1][0]),
      's_c3': np.std(cmp_dwt[2][0])
  }
  return vector

def img2vec(filename, dim):
  image = load_image(filename)
  if image.shape[:2] != dim:
    tempname = "temp" + splitext(filename)[1].lower()
    save_image(resize_image(image, dim), tempname)
    image = load_image(tempname)
    os.remove(tempname)
  rgb_channels = get_rgb_channels(image)
  rgb_dwt = get_dwt(rgb_channels, 5)
  cmp_channels = get_cmp_channels(image)
  cmp_dwt = get_dwt(cmp_channels, 5)
  return get_feature_vector(cmp_dwt, rgb_dwt)

def img2bytes(image):
  return cv2.imencode(".png", image)[1].tobytes()