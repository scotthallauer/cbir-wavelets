import numpy as np
import cv2
import pywt

def load_image(filename):
  return cv2.imread(filename)

def save_image(image, filename):
  cv2.imwrite(filename, image)

def resize_image(image, dim):
  return cv2.resize(image, dim, interpolation = cv2.INTER_LINEAR)

def get_rgb(image):
  b, g, r = cv2.split(image)
  return (r, g, b)

def get_icc(image):
  cmp_max = 255
  r, g, b = get_rgb(image)
  i = (r + g + b) / 3 # intensity
  c1 = (r + (cmp_max - b)) / 2 # contrast 1
  c2 = (r + 2 * (cmp_max - g) + b) / 4 # contrast 2
  return (i, c1, c2)

def get_dwt(components, level):
  c1, c2, c3 = components
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

def get_feature_vector(dwt):
  w_c1, w_c2, w_c3 = dwt
  vector = {
      'w_c1': w_c1,
      'w_c2': w_c2,
      'w_c3': w_c3,
      's_c1': np.std(w_c1[0]),
      's_c2': np.std(w_c2[0]),
      's_c3': np.std(w_c3[0])
  }
  return vector

def img2vec(filename, dim):
  image = load_image(filename)
  if image.shape[:2] != dim:
    image = resize_image(image, dim)
  components = get_icc(image)
  dwt = get_dwt(components, 5)
  return get_feature_vector(dwt)

def img2bytes(image):
  return cv2.imencode(".png", image)[1].tobytes()