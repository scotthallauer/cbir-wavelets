import numpy as np
import os
import cv2
import pywt
import pickle

component_max = 255
image_dim = (128, 128)

def resize_image(original, dim):
  return cv2.resize(original, dim, interpolation = cv2.INTER_LINEAR)

def get_rgb(image):
  b, g, r = cv2.split(image)
  return (r, g, b)

def get_components(rgb):
  r, g, b = rgb
  c1 = (r + g + b) / 3
  c2 = (r + (component_max - b)) / 2
  c3 = (r + 2 * (component_max - g) + b) / 4
  return (c1, c2, c3)

def get_dwt(components, level):
  c1, c2, c3 = components
  coeff_c1 = pywt.dwt2(c1, 'bior1.3')
  coeff_c2 = pywt.dwt2(c2, 'bior1.3')
  coeff_c3 = pywt.dwt2(c3, 'bior1.3')
  for i in range(level-1):
    coeff_c1 = pywt.dwt2(coeff_c1[0], 'bior1.3')
    coeff_c2 = pywt.dwt2(coeff_c2[0], 'bior1.3')
    coeff_c3 = pywt.dwt2(coeff_c3[0], 'bior1.3')
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

def img2vec(filename):
  image = cv2.imread(filename)
  image = resize_image(image, image_dim)
  rgb = get_rgb(image)
  components = get_components(rgb)
  dwt = get_dwt(components, 5)
  return get_feature_vector(dwt)

def pickle_dump(database, path):
  with open(path, 'wb') as fo:
    pickle.dump(database, fo, protocol=pickle.HIGHEST_PROTOCOL)

def pickle_load(file):
  with open(file, 'rb') as fo:
    database = pickle.load(fo)
  return database

def batch_resize(src, dest):
  files = [f for f in os.listdir(src) if os.path.isfile(os.path.join(src, f)) and f != '.DS_Store']
  print(f'Processing {len(files)} images...')
  for f in files:
    try:
      image = cv2.imread(os.path.join(src, f))
      image = resize_image(image, image_dim)
      cv2.imwrite(os.path.join(dest, f), image)
    except:
      print(f'Resizing \'{f}\' failed.')
  print(f'Complete.')

def batch_vectorize(src, dest):
  files = [f for f in os.listdir(src) if os.path.isfile(os.path.join(src, f)) and f != '.DS_Store']
  print(f'Processing {len(files)} images...')
  database = {}
  database["size"] = 0
  database["image"] = []
  for f in files:
    try:
      vector = img2vec(os.path.join(src, f))
      database["size"] += 1
      database["image"].append({
        "file": f,
        "vector": vector
      })
    except:
      print(f'Vectorise \'{f}\' failed.')
  pickle_dump(database, dest)
  print(f'Complete.')

def load_database(path):
  return pickle_load(path)