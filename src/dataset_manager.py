import copy
import dataset_processor as dp
from os import listdir, mkdir
from os.path import join, isfile, isdir

class DatasetError(Exception):
  """A custom exception used to report errors in use of DatasetManager class"""

class DatasetManager:
  def __init__(self, root_path, image_dim):
    self._data_path = join(root_path, "data")
    if not isdir(self._data_path):
      try:
        mkdir(self._data_path)
      except:
        raise DatasetError(f"Application data path '{self._data_path}' does not exist and could not be created.")
    self._image_dim = image_dim
    self._datasets = []
    self._loaded_dataset = {
      "id": None,
      "path": None,
      "title": None,
      "database": None
    }
    self._active = False
    self._loaded = False

  def reset(self):
    if not isdir(self._data_path):
      try:
        mkdir(self._data_path)
      except:
        raise DatasetError(f"Application data path '{self._data_path}' does not exist and could not be created.")
    self._datasets.clear()
    self._loaded_dataset = {
      "id": None,
      "path": None,
      "title": None,
      "database": None
    }
    self._active = False
    self._loaded = False

  def discover_datasets(self):
    self.reset()
    dirs = [d for d in listdir(self._data_path) if isdir(join(self._data_path, d)) and d.startswith("dataset")]
    for d in dirs:
      dataset_id = d[7:]
      dataset_path = join(self._data_path, d)
      dataset_title = f"Dataset {dataset_id}"
      title_path = join(dataset_path, "title.txt")
      if isfile(title_path):
        with open(title_path) as fo:
          dataset_title = fo.readline().rstrip()
      self._datasets.append({"id": dataset_id, "path": dataset_path, "title": dataset_title})
    self._datasets = sorted(self._datasets, key=lambda r: r["title"])
    self._active = True
    print(f"Discovered {len(self._datasets)} dataset(s).")

  def load_dataset(self, dataset_id):
    if not self._active:
      self.discover_datasets()
    dataset_id = str(dataset_id)
    for d in self._datasets:
      if d["id"] == dataset_id:
        self._loaded_dataset = {
          "id": d["id"],
          "path": d["path"],
          "title": d["title"],
          "database": dp.load_database(join(d["path"], "database.pickle"))
        }
        self._loaded = True
        print(f"Loaded dataset '{self._loaded_dataset['title']}' with {self._loaded_dataset['database']['size']} image(s).")
        return
    raise DatasetError(f"Cannot load dataset with ID '{dataset_id}' because no matching dataset exists.")

  def import_dataset(self, src, title=""):
    if not isdir(src):
      raise DatasetError(f"Cannot import dataset because the source path '{src}' does not exist or is not a directory.")
    dataset_id = self.next_id()
    dataset_title = title if len(title) > 0 else f"Dataset {dataset_id}"
    dataset_path = join(self._data_path, f"dataset{dataset_id}")
    mkdir(dataset_path)
    if len(title) > 0:
      with open(join(dataset_path, "title.txt"), "w") as fo:
        fo.write(title)
    dirs = [src] + [join(src, d) for d in listdir(src) if isdir(join(src, d))]
    print(f"Processing {len(dirs)} folder(s)...")
    copy_time = 0
    file_idx = 1
    for d in dirs:
      print(f"Processing folder '{d}'...")
      time, file_idx = dp.batch_copy(d, join(dataset_path, "original"), file_idx)
      copy_time += time
    resize_time = dp.batch_resize(join(dataset_path, "original"), join(dataset_path, "resized"), self._image_dim)
    vectorize_time = dp.batch_vectorize(join(dataset_path, "resized"), join(dataset_path, "database.pickle"), self._image_dim)
    print(f"Imported dataset '{dataset_title}'.")
    self.discover_datasets()
    self.load_dataset(dataset_id)
    return {"c": copy_time, "r": resize_time, "v": vectorize_time}

  def list_datasets(self):
    if not self._active:
      self.discover_datasets()
    return copy.deepcopy(self._datasets)

  def get_id(self):
    if not self._loaded:
      raise DatasetError(f"Cannot return dataset ID because no dataset has been loaded.")
    return copy.deepcopy(self._loaded_dataset["id"])

  def get_title(self):
    if not self._loaded:
      raise DatasetError(f"Cannot return dataset title because no dataset has been loaded.")
    return copy.deepcopy(self._loaded_dataset["title"])

  def get_path(self):
    if not self._loaded:
      raise DatasetError(f"Cannot return dataset path because no dataset has been loaded.")
    return copy.deepcopy(self._loaded_dataset["path"])

  def database(self):
    if not self._loaded:
      raise DatasetError(f"Cannot return dataset database because no dataset has been loaded.")
    return self._loaded_dataset["database"]

  def is_loaded(self, dataset_id):
    return self._loaded_dataset["id"] == str(dataset_id)

  def exists(self, dataset_id):
    if not self._active:
      self.discover_datasets()
    dataset_id = str(dataset_id)
    for d in self._datasets:
      if d["id"] == dataset_id:
        return True
    return False

  def next_id(self):
    if not self._active:
      self.discover_datasets()
    dataset_id = 1
    while self.exists(dataset_id):
      dataset_id += 1
    return str(dataset_id)