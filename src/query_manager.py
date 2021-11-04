import copy
import cv2
import math
import matplotlib.pyplot as plt
import image_processor as ip
import image_comparator as ic
from os import listdir, mkdir
from os.path import join, isfile, isdir
from timer import Timer

class QueryError(Exception):
  """A custom exception used to report errors in use of QueryManager class"""

class QueryManager:
  def __init__(self, root_path, dataset_manager, image_dim):
    self._results_path = join(root_path, "results")
    self._dataset_manager = dataset_manager
    self._image_dim = image_dim
    self._timer = Timer()
    self._executed = False
    self._dataset = {
      "id": None,
      "path": None,
      "title": None
    }
    self._query = None
    self._results = []

  def process_query(self, query):
    self._timer.start()
    self._dataset = {
      "id": self._dataset_manager.get_id(),
      "path": self._dataset_manager.get_path(),
      "title": self._dataset_manager.get_title()
    }
    self._query = copy.deepcopy(query)
    self._results.clear()
    query_vector = ip.img2vec(self._query["image"]["path"], self._image_dim)
    progress = 0
    for i, candidate in enumerate(self._dataset_manager.database()["image"]):
      passed, score = ic.pair2score(query_vector, candidate["vector"], self._query["params"])
      if passed:
        self._results.append({
          "file": candidate["file"],
          "score": score
        })
      new_progress = round((i/self._dataset_manager.database()["size"])*10)*10
      if progress != new_progress:
        progress = new_progress
        print(f"{progress}%")
    self._results = sorted(self._results, key=lambda r: r["score"])
    self._timer.stop()
    self._executed = True
    print(f"""Results: {len(self._results)} image(s){f' (with best match distance {round(self._results[0]["score"])})' if len(self._results) > 0 else ''}""")
    print(f"Time: {'{:.2f}'.format(self._timer.time())} seconds")

  def get_results(self):
    if not self._executed:
      raise QueryError(f"Cannot return query results because no query has been executed.")
    return self._results

  def get_time(self):
    if not self._executed:
      raise QueryError(f"Cannot return query time because no query has been executed.")
    return self._timer.time()

  def export_results(self):
    num_images = min(len(self._results), self._query["params"]["limit"]) + 1
    images = [self._query["image"]["large"]] + [ip.resize_image(cv2.imread(join(self._dataset["path"], "original", r["file"])), (228,228)) for r in self._results[:(num_images-1)]]
    if num_images > 0:
      cols = 5
      rows = math.ceil(num_images/cols)
      fig, axs = plt.subplots(rows, cols, figsize=(cols*3,rows*3), dpi=100)
      for i in range(rows*cols):
        ax_idx = (math.floor(i/5), i%5) if rows > 1 else i%5
        if i >= num_images:
          axs[ax_idx].axis('off')
        else:
          if i == 0:
            [x.set_linewidth(2) for x in axs[ax_idx].spines.values()]
          axs[ax_idx].imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB), interpolation="bilinear", cmap=plt.cm.gray)
          axs[ax_idx].set_title((f"Result {i}" if i > 0 else "Query"), fontsize=10, fontweight=("normal" if i > 0 else "bold"))
          axs[ax_idx].set_xticks([])
          axs[ax_idx].set_yticks([])
      plt.figtext(0.5, 0.07, f"Dataset: {self._dataset['title']}\nQuery Parameters: {str(self._query['params'])}", wrap=True, horizontalalignment="center", fontsize=12)
      idx = 1
      if not isdir(self._results_path):
        mkdir(self._results_path)
      while isfile(join(self._results_path, f"results{idx}.png")):
        idx += 1
      filename = join(self._results_path, f"results{idx}.png")
      plt.savefig(filename, bbox_inches="tight")
      return filename