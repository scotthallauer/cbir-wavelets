import pickle
import image_processor as ip
from os import listdir
from os.path import join, isdir, isfile
from timer import Timer

t = Timer()

def batch_resize(src, dst, dim):
  t.start()
  files = [f for f in listdir(src) if isfile(join(src, f))]
  print(f'Processing {len(files)} files...')
  for f in files:
    try:
      image = ip.load_image(join(src, f))
      image = ip.resize_image(image, dim)
      ip.save_image(image, join(dst, f))
    except:
      print(f'Resizing \'{f}\' failed.')
  print(f'Complete.')
  t.stop()
  return t.time()

def batch_vectorize(src, filename, dim):
  t.start()
  files = [f for f in listdir(src) if isfile(join(src, f))]
  print(f'Processing {len(files)} files...')
  database = {}
  database["size"] = 0
  database["image"] = []
  for f in files:
    try:
      vector = ip.img2vec(join(src, f), dim)
      database["size"] += 1
      database["image"].append({
        "file": f,
        "vector": vector
      })
    except:
      print(f'Vectorizing \'{f}\' failed.')
  pickle_dump(database, filename)
  print(f'Complete.')
  t.stop()
  return t.time()

def pickle_dump(database, filename):
  with open(filename, 'wb') as fo:
    pickle.dump(database, fo, protocol=pickle.HIGHEST_PROTOCOL)

def pickle_load(filename):
  with open(filename, 'rb') as fo:
    database = pickle.load(fo)
  return database

def load_database(filename):
  return pickle_load(filename)