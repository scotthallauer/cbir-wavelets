import dataset_processor as dp
from os import listdir, mkdir
from os.path import join, isfile, isdir

class DatasetError(Exception):
  """A custom exception used to report errors in use of DatasetManager class"""

class DatasetManager:
  def __init__(self, path, image_dim):
    if not isdir(path):
      try:
        mkdir(path)
      except:
        raise DatasetError(f"Application data path '{path}' does not exist and could not be created.")
    self._path = path
    self._image_dim = image_dim
    self._active = False
    self._loaded = False
    self._datasets = []
    self._selected_idx = None
    self.database = {}

  def reset(self):
    if not isdir(self._path):
      try:
        mkdir(self._path)
      except:
        raise DatasetError(f"Application data path '{self._path}' does not exist and could not be created.")
    self._active = False
    self._loaded = False
    self._datasets.clear()
    self._selected_idx = None
    self.database = {}

  def discover_datasets(self):
    self.reset()
    dirs = [d for d in listdir(self._path) if isdir(join(self._path, d)) and d.startswith("dataset")]
    for d in dirs:
      idx = d[7:]
      title_path = join(self._path, d, "title.txt")
      if isfile(title_path):
        with open(title_path) as fo:
          title = fo.readline().rstrip()
      else:
        title = f"Dataset {idx}"
      self._datasets.append({"idx": idx, "title": title})
    self._datasets = sorted(self._datasets, key=lambda r: r["title"])
    self._active = True
    print(f"Discovered {len(self._datasets)} dataset(s).")

  def load_dataset(self, idx):
    if not self._active:
      self.discover_datasets()
    idx = str(idx)
    if self.exists(idx):
      self._selected_idx = idx
      self.database = dp.load_database(join(self._path, f"dataset{idx}", "database.pickle"))
      self._loaded = True
      print(f"Loaded dataset '{self.get_title()}' with {self.database['size']} image(s).")

  def import_dataset(self, src, title=""):
    if not isdir(src):
      raise DatasetError(f"Dataset could not be imported because the source path '{src}' does not exist or is not a directory.")
    idx = self.next_idx()
    title = title if len(title) > 0 else f"Dataset {idx}"
    dst = join(self._path, f"dataset{idx}")
    mkdir(dst)
    if len(title) > 0:
      with open(join(dst, "title.txt"), "w") as fo:
        fo.write(title)
    dirs = [src] + [join(src, d) for d in listdir(src) if isdir(join(src, d))]
    print(f"Processing {len(dirs)} folder(s)...")
    copy_time = 0
    file_idx = 1
    for d in dirs:
      print(f"Processing folder '{d}'...")
      c, i = dp.batch_copy(d, join(dst, "original"), file_idx)
      file_idx = i
      copy_time += c
    resize_time = dp.batch_resize(join(dst, "original"), join(dst, "resized"), self._image_dim)
    vectorize_time = dp.batch_vectorize(join(dst, "resized"), join(dst, "database.pickle"), self._image_dim)
    print(f"Imported dataset '{title}'.")
    self.discover_datasets()
    self.load_dataset(idx)
    return {"c": copy_time, "r": resize_time, "v": vectorize_time}

  def get_datasets(self):
    if not self._active:
      self.discover_datasets()
    return self._datasets

  def get_title(self, idx=None):
    if idx is None:
      idx = self._selected_idx
    idx = str(idx)
    if self.exists(idx):
      for d in self._datasets:
        if d["idx"] == idx:
          return d["title"]
    return None

  def get_path(self, idx=None):
    if idx is None:
      idx = self._selected_idx
    idx = str(idx)
    return join(self._path, f"dataset{idx}")

  def is_selected(self, idx):
    idx = str(idx)
    return self._selected_idx == idx

  def exists(self, idx):
    if not self._active:
      self.discover_datasets()
    idx = str(idx)
    for d in self._datasets:
      if d["idx"] == idx:
        return True
    return False

  def next_idx(self):
    if not self._active:
      self.discover_datasets()
    idx = 1
    while self.exists(idx):
      idx += 1
    return str(idx)