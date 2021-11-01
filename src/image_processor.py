import numpy as np
import cv2
import pywt
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
  return (r, g, b)

# Component colour space
def get_cmp_channels(image):
  cmp_max = 255
  r, g, b = get_rgb_channels(image)
  i = (r + g + b) / 3 # intensity
  c1 = (r + (cmp_max - b)) / 2 # contrast 1
  c2 = (r + 2*(cmp_max - g) + b) / 4 # contrast 2
  return (i, c1, c2)

# Opponent colour space
def get_opp_channels(image):
  r, g, b = get_rgb_channels(image)
  rg = r - 2*g + b
  by = -r - g + 2*b
  wb = r + g + b
  return (rg, by, wb)

def get_dwt(channels, level):
  c1, c2, c3 = channels
  coeff_c1 = pywt.dwt2(c1, 'bior1.3')
  coeff_c2 = pywt.dwt2(c2, 'bior1.3')
  coeff_c3 = pywt.dwt2(c3, 'bior1.3')
  for i in range(level-1):
    coeff_c1 = pywt.dwt2(coeff_c1[0], 'bior1.3')
    coeff_c1 = np.array([coeff_c1[0], coeff_c1[1][0], coeff_c1[1][1], coeff_c1[1][2]])
    coeff_c2 = pywt.dwt2(coeff_c2[0], 'bior1.3')
    coeff_c2 = np.array([coeff_c2[0], coeff_c2[1][0], coeff_c2[1][1], coeff_c2[1][2]])
    coeff_c3 = pywt.dwt2(coeff_c3[0], 'bior1.3')
    coeff_c3 = np.array([coeff_c3[0], coeff_c3[1][0], coeff_c3[1][1], coeff_c3[1][2]])
  return (coeff_c1, coeff_c2, coeff_c3)

def get_feature_vector(cmp_dwt, rgb_dwt):
  w_c1, w_c2, w_c3 = cmp_dwt
  w_c4, w_c5, w_c6 = rgb_dwt
  vector = {
      'w_c1': w_c1,
      'w_c2': w_c2,
      'w_c3': w_c3,
      'w_c4': w_c4,
      'w_c5': w_c5,
      'w_c6': w_c6,
      's_c1': np.std(w_c1[0]),
      's_c2': np.std(w_c2[0]),
      's_c3': np.std(w_c3[0])
  }
  return vector

def img2vec(filename, dim):
  image = load_image(filename)
  if image.shape[:2] != dim:
    image = resize_image(image, dim)
  rgb_channels = get_rgb_channels(image)
  rgb_dwt = get_dwt(rgb_channels, 5)
  cmp_channels = get_cmp_channels(image)
  cmp_dwt = get_dwt(cmp_channels, 5)
  return get_feature_vector(cmp_dwt, rgb_dwt)

def img2bytes(image):
  return cv2.imencode(".png", image)[1].tobytes()