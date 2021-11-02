import pickle
import image_processor as ip
from os import listdir, mkdir
from os.path import join, isdir, isfile, splitext
from shutil import copyfile
from timer import Timer

t = Timer()

def batch_copy(src, dst, idx):
  t.start()
  files = [f for f in listdir(src) if isfile(join(src, f)) and ip.is_supported(f)]
  if not isdir(dst):
    mkdir(dst)
  print(f"Copying {len(files)} files...")
  for f in files:
    try:
      copyfile(join(src, f), join(dst, f"image{idx}{splitext(f)[1].lower()}"))
      idx += 1
    except:
      print(f"Copying '{f}' failed.")
  t.stop()
  print(f"Complete ({'{:.2f}'.format(t.time())} seconds).")
  return (t.time(), idx)

def batch_resize(src, dst, dim):
  t.start()
  files = [f for f in listdir(src) if isfile(join(src, f)) and ip.is_supported(f)]
  if not isdir(dst):
    mkdir(dst)
  print(f"Resizing {len(files)} files...")
  progress = 0
  for idx, f in enumerate(files):
    try:
      image = ip.load_image(join(src, f))
      image = ip.resize_image(image, dim)
      ip.save_image(image, join(dst, f))
    except:
      print(f"Resizing '{f}' failed.")
    new_progress = round((idx/len(files))*10)*10
    if progress != new_progress:
      progress = new_progress
      print(f"{progress}%")
  t.stop()
  print(f"Complete ({'{:.2f}'.format(t.time())} seconds).")
  return t.time()

def batch_vectorize(src, filename, dim):
  t.start()
  files = [f for f in listdir(src) if isfile(join(src, f))]
  print(f"Vectorizing {len(files)} files...")
  database = {}
  database["size"] = 0
  database["image"] = []
  progress = 0
  for idx, f in enumerate(files):
    try:
      vector = ip.img2vec(join(src, f), dim)
      database["size"] += 1
      database["image"].append({
        "file": f,
        "vector": vector
      })
    except:
      print(f"Vectorizing '{f}' failed.")
    new_progress = round((idx/len(files))*10)*10
    if progress != new_progress:
      progress = new_progress
      print(f"{progress}%")
  pickle_dump(database, filename)
  t.stop()
  print(f"Complete ({'{:.2f}'.format(t.time())} seconds).")
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