import PySimpleGUI as sg
import cv2
import math
import matplotlib.pyplot as plt
import dataset_processor as dp
import image_processor as ip
import image_comparator as ic
from os import mkdir, getcwd
from os.path import join, isfile, isdir
from timer import Timer

# WINDOW SETTINGS

WINDOW_DIM = (1200, 700)

def width(proportion):
  return math.floor(WINDOW_DIM[0] * proportion)

def height(proportion):
  return math.floor(WINDOW_DIM[1] * proportion)

# GLOBAL PARAMETERS

ROOT = getcwd()

IMAGE_DIM = (128, 128)

DATABASE = None

SELECTED_DATASET = None

DATASETS = []

WINDOW_OPENED = False

T = Timer()

QUERY = {
  "image": {
    "path": join(ROOT, "src/media/query/query1.jpg"),
    "small": None,
    "large": None
  },
  "params": {
    "percent": 50,
    "threshold": 100000,
    "w_quad": [1,1,1,1],
    "w_comp": [1,2,2],
    "limit": 20
  }
}

RESULTS = []

BLANK_IMAGE = ip.img2bytes(ip.resize_image(cv2.imread(join(ROOT, "src/media/blank.png")), (width(0.137), width(0.137))))

# FUNCTIONS

def load_query_image(values):
  QUERY["image"]["path"] = values["QUERY_PATH"]
  image = cv2.imread(QUERY["image"]["path"])
  QUERY["image"]["small"] = ip.resize_image(image, IMAGE_DIM)
  QUERY["image"]["large"] = ip.resize_image(image, (width(0.19), width(0.19)))

def display_query_image():
  WINDOW["QUERY_IMAGE"].update(data=ip.img2bytes(QUERY["image"]["large"]))

def load_query_param(values):
  QUERY["params"] = {
    "percent": int(values["PARAM_PERCENT"]),
    "threshold": int(values["PARAM_THRESHOLD"]),
    "w_quad": [
      int(values["PARAM_W11"]),
      int(values["PARAM_W12"]),
      int(values["PARAM_W21"]),
      int(values["PARAM_W22"])
    ],
    "w_comp": [
      int(values["PARAM_WC1"]),
      int(values["PARAM_WC2"]),
      int(values["PARAM_WC3"])
    ],
    "limit": int(values["PARAM_LIMIT"])
  }

def update_query(values):
  load_query_image(values)
  display_query_image()
  load_query_param(values)
  print(QUERY["params"])

def process_query():
  global RESULTS
  T.start()
  RESULTS.clear()
  query_vector = ip.img2vec(QUERY["image"]["path"], IMAGE_DIM)
  for candidate in DATABASE["image"]:
    passed, score = ic.pair2score(query_vector, candidate["vector"], QUERY["params"])
    if passed:
      RESULTS.append({
        "image": ip.resize_image(cv2.imread(join(ROOT, "data", f"dataset{SELECTED_DATASET}", "original", candidate['file'])), (width(0.137), width(0.137))),
        "score": score
      })
  RESULTS = sorted(RESULTS, key=lambda r: r["score"])
  T.stop()

def display_results():
  for i in range(50):
    if i < min(len(RESULTS), QUERY["params"]["limit"]):
      WINDOW[f"RESULT_IMAGE_{i}"].update(data=ip.img2bytes(RESULTS[i]["image"]))
      WINDOW[f"RESULT_IMAGE_{i}"].set_tooltip(str(RESULTS[i]["score"]))
    else:
      WINDOW[f"RESULT_IMAGE_{i}"].update(data=BLANK_IMAGE)
  if len(RESULTS) == 0:
    WINDOW["_EXPORT_"].update(visible=False)
  else:
    WINDOW["_EXPORT_"].update(visible=True)

def clear_results():
  global RESULTS
  RESULTS.clear()
  for i in range(50):
    WINDOW[f"RESULT_IMAGE_{i}"].update(data=BLANK_IMAGE)
  WINDOW["_EXPORT_"].update(visible=False)
  WINDOW["_STATS_"].update(visible=False)

def export_results():
  num_images = min(len(RESULTS), QUERY["params"]["limit"]) + 1
  IMAGES = [QUERY["image"]["large"]] + [r["image"] for r in RESULTS[:(num_images-1)]]
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
        axs[ax_idx].imshow(cv2.cvtColor(IMAGES[i], cv2.COLOR_BGR2RGB), interpolation="bilinear", cmap=plt.cm.gray)
        axs[ax_idx].set_title((f"Result {i}" if i > 0 else "Query"), fontsize=10, fontweight=("normal" if i > 0 else "bold"))
        axs[ax_idx].set_xticks([])
        axs[ax_idx].set_yticks([])
    plt.figtext(0.5, 0.07, f"Dataset: {get_dataset_title(SELECTED_DATASET)}\nQuery Parameters: {str(QUERY['params'])}", wrap=True, horizontalalignment="center", fontsize=12)
    idx = 1
    results_path = join(ROOT, "results")
    if not isdir(results_path):
      mkdir(results_path)
    while isfile(join(results_path, f"results{idx}.png")):
      idx += 1
    plt.savefig(join(results_path, f"results{idx}.png"), bbox_inches="tight")
    return f"results{idx}.png"

def display_stats():
  WINDOW["_STATS_"].update(
    f"Showing {min(len(RESULTS), QUERY['params']['limit'])} of {len(RESULTS)} results ({'{:.2f}'.format(T.time())} seconds)", 
    visible=True)

def find_datasets():
  global DATASETS
  data_path = join(ROOT, "data")
  if not isdir(data_path):
    mkdir(data_path)
  idx = 1
  DATASETS.clear()
  while isdir(join(data_path, f"dataset{idx}")):
    title_path = join(data_path, f"dataset{idx}", "title.txt")
    if isfile(title_path):
      with open(title_path) as fo:
        title = fo.readline().rstrip()
    else:
      title = f"Dataset {idx}"
    DATASETS.append({"title": title, "idx": idx})
    idx += 1
  if SELECTED_DATASET is None and len(DATASETS) > 0:
    load_dataset(1)

def import_dataset():
  input_left_column = [
    [
      sg.Text("Title")
    ],
    [
      sg.Text("Path *")
    ]
  ]
  input_right_column = [
    [
      sg.Input("", size=(35, 1), enable_events=True, key="_NEW_DATASET_TITLE_")
    ],
    [
      sg.Input("", size=(26, 1), enable_events=True, key="_NEW_DATASET_PATH_"),
      sg.FolderBrowse()
    ]
  ]
  layout = [
    [sg.Text("To import a dataset, please provide a directory containing only\nimages. These images will be copied into the application and\ntheir feature vectors will be extracted.")],
    [sg.Text("You can also optionally provide a title for your dataset.")],
    [sg.Frame(title="Dataset Information", font=("Helvetica", 11), pad=(10, 10), layout=[[sg.Column(input_left_column, pad=(10, 10)), sg.Column(input_right_column, pad=(10, 10))]])],
    [sg.Button("Import", pad=(10, 5)), sg.Button("Cancel")]
  ]
  window = sg.Window("Import Dataset", layout, modal=True, keep_on_top=True)
  choice = None
  title, src = "", ""
  while True:
    event, values = window.read()
    if event == "Exit" or event == "Cancel" or event == sg.WIN_CLOSED:
      if not WINDOW_OPENED:
        exit()
      window.close()
      break
    if event == "Import":
      find_datasets()
      title = values["_NEW_DATASET_TITLE_"]
      src = values["_NEW_DATASET_PATH_"]
      if isdir(src):
        idx = len(DATASETS) + 1
        dataset_path = join(ROOT, "data", f"dataset{idx}")
        mkdir(dataset_path)
        if len(title) > 0:
          with open(join(dataset_path, "title.txt"), "w") as fo:
            fo.write(title)
        copy_time = dp.batch_copy(src, join(dataset_path, "original"))
        resize_time = dp.batch_resize(join(dataset_path, "original"), join(dataset_path, "resized"), (128, 128))
        vectorize_time = dp.batch_vectorize(join(dataset_path, "resized"), join(dataset_path, "database.pickle"), (128, 128))
        window.close()
        find_datasets()
        load_dataset(idx)
        if WINDOW_OPENED:
          WINDOW["_MENU_"].update(menu_definition=generate_menu())
          WINDOW["_DATASET_"].update(get_dataset_title(idx))
        sg.Popup(
          f"The dataset '{get_dataset_title(idx)}' has been imported successfully.\n\n" +
          f"Size: {DATABASE['size']} images\n\n" +
          f"Copy Time: {'{:.2f}'.format(copy_time)} seconds\n" +
          f"Resize Time: {'{:.2f}'.format(resize_time)} seconds\n" +
          f"Vectorize Time: {'{:.2f}'.format(vectorize_time)} seconds\n", 
          title="Dataset Imported", 
          keep_on_top=True
        )
        break
  

def load_dataset(idx):
  global SELECTED_DATASET, DATABASE
  SELECTED_DATASET = idx
  DATABASE = dp.load_database(join(ROOT, "data", f"dataset{idx}", "database.pickle"))

def get_dataset_title(idx):
  for d in DATASETS:
    if d["idx"] == idx:
      return d["title"]

def generate_menu():
  menu_datasets = []
  for d in DATASETS:
    menu_datasets.append(f"{'!' if d['idx'] == SELECTED_DATASET else ''}{d['title']}{' (selected)' if d['idx'] == SELECTED_DATASET else ''}::_DATASET-{d['idx']}_")
  return ['&File', ['&Import Dataset...', '&Select Dataset', menu_datasets]], ['&Edit']

# INITIALISE

find_datasets()

if len(DATASETS) == 0:
  import_dataset()

load_query_image({"QUERY_PATH": QUERY["image"]["path"]})

MENU = generate_menu()

# WINDOW LAYOUT

MATRIX_WEIGHT_LEFT_COLUMN = [
  [sg.Slider(default_value=QUERY["params"]["w_quad"][0], range=(0,5), size=(17,15), pad=(5, (5, 0)), orientation="h", key="PARAM_W11")],
  [sg.Text("w11", pad=(5, (0, 5)), justification="center", expand_x=True)],
  [sg.Slider(default_value=QUERY["params"]["w_quad"][2], range=(0,5), size=(17,15), pad=(5, (5, 0)), orientation="h", key="PARAM_W21")],
  [sg.Text("w21", pad=(5, (0, 5)), justification="center", expand_x=True)],
]

MATRIX_WEIGHT_RIGHT_COLUMN = [
  [sg.Slider(default_value=QUERY["params"]["w_quad"][1], range=(0,5), size=(17,15), pad=(5, (5, 0)), orientation="h", key="PARAM_W12")],
  [sg.Text("w12", pad=(5, (0, 5)), justification="center", expand_x=True)],
  [sg.Slider(default_value=QUERY["params"]["w_quad"][3], range=(0,5), size=(17,15), pad=(5, (5, 0)), orientation="h", key="PARAM_W22")],
  [sg.Text("w22", pad=(5, (0, 5)), justification="center", expand_x=True)],
]

COMPONENT_WEIGHT_FRAME = [
  [sg.Slider(default_value=QUERY["params"]["w_comp"][0], range=(0,5), size=(17,15), pad=(5, (5, 0)), orientation="h", key="PARAM_WC1")],
  [sg.Text("wc1", pad=(5, (0, 5)), justification="center", expand_x=True)],
  [sg.Slider(default_value=QUERY["params"]["w_comp"][1], range=(0,5), size=(17,15), pad=(5, (5, 0)), orientation="h", key="PARAM_WC2")],
  [sg.Text("wc2", pad=(5, (0, 5)), justification="center", expand_x=True)],
  [sg.Slider(default_value=QUERY["params"]["w_comp"][2], range=(0,5), size=(17,15), pad=(5, (5, 0)), orientation="h", key="PARAM_WC3")],
  [sg.Text("wc3", pad=(5, (0, 5)), justification="center", expand_x=True)],
]

SELECTION_SETTING_FRAME = [
  [sg.Slider(default_value=QUERY["params"]["percent"], range=(1,100), size=(17,15), pad=(5, (5, 0)), orientation="h", key="PARAM_PERCENT")],
  [sg.Text("percent", pad=(5, (0, 5)), justification="center", expand_x=True)],
  [sg.Slider(default_value=QUERY["params"]["threshold"], range=(50000,1000000), size=(17,15), pad=(5, (5, 0)), orientation="h", key="PARAM_THRESHOLD")],
  [sg.Text("threshold", pad=(5, (0, 5)), justification="center", expand_x=True)],
  [sg.Slider(default_value=QUERY["params"]["limit"], range=(1,50), size=(17,15), pad=(5, (5, 0)), orientation="h", key="PARAM_LIMIT")],
  [sg.Text("max results", pad=(5, (0, 5)), justification="center", expand_x=True)],
]

QUERY_COLUMN = [
  [
    sg.Text("Query", font=("Helvetica", 20, "bold")),
  ],
  [
    sg.Text("Selected Dataset:", font=("Helvetica", 10, "bold")),
    sg.Text(get_dataset_title(SELECTED_DATASET), pad=((5, 0), 5), key="_DATASET_"),
    sg.Text("[?]", font=("Helvetica", 8), pad=((2, 5), 5), tooltip="To change dataset, go to File > Select Dataset")
  ],
  [
    sg.Image(data=ip.img2bytes(QUERY["image"]["large"]), pad=(width(0.03), 5), key="QUERY_IMAGE")
  ],
  [
    sg.Text("Path"),
    sg.Input(QUERY["image"]["path"], size=(23, 1), enable_events=True, key="QUERY_PATH"),
    sg.FileBrowse(initial_folder=join(ROOT, "src/media/query")),
    sg.Button("Load")
  ],
  [
    sg.Frame(title="Matrix Weights", font=("Helvetica", 11), layout=[[sg.Column(MATRIX_WEIGHT_LEFT_COLUMN), sg.Column(MATRIX_WEIGHT_RIGHT_COLUMN)]])
  ],
  [
    sg.Frame(title="Channel Weights", font=("Helvetica", 11), pad=((5, 8), 5), layout=COMPONENT_WEIGHT_FRAME),
    sg.Frame(title="Selection Settings", font=("Helvetica", 11), pad=((8, 5), 5), layout=SELECTION_SETTING_FRAME),
  ],
  [
    sg.Button("Search", pad=((5, 12), 10), expand_x=True)
  ]
]

RESULTS_COLUMN = [
  [
    sg.Text("Results", font=("Helvetica", 20, "bold"), key="RESULTS"),
    sg.Button("Export", visible=False, key="_EXPORT_")
  ],
  [
    sg.Text("", key="_STATS_", visible=False)
  ]
]

for row in range(10):
  result_images = []
  for col in range(5):
    result_images.append(sg.Image(data=BLANK_IMAGE, key=f"RESULT_IMAGE_{row*5 + col}"))
  RESULTS_COLUMN.append(result_images)


LAYOUT = [
  [sg.Menu(MENU, font='Helvetica', pad=(10,10), key="_MENU_")],
  [
    sg.Column(QUERY_COLUMN, size=(width(0.25), height(1)), scrollable=True),
    sg.VSeparator(),
    sg.Column(RESULTS_COLUMN, size=(width(0.75), height(1)), scrollable=True, vertical_scroll_only=True),
  ]
]

# RUN

WINDOW = sg.Window("Wavelet CBIR Search Engine", LAYOUT)

WINDOW_OPENED = True

while True:
  event, values = WINDOW.read()
  if event == "Exit" or event == sg.WIN_CLOSED:
    break
  if event == "Load":
    load_query_image(values)
    display_query_image()
  if event == "Search":
    update_query(values)
    process_query()
    display_results()
    display_stats()
  if event == "_EXPORT_":
    filename = export_results()
    sg.Popup(f"Results exported to 'results/{filename}'.", title="Export Complete", keep_on_top=True)
  if event == "Import Dataset...":
    import_dataset()
  if "_DATASET-" in event:
    idx = int(event[event.index("_DATASET-") + 9:len(event)-1])
    title = get_dataset_title(idx)
    load_dataset(idx)
    clear_results()
    WINDOW["_MENU_"].update(menu_definition=generate_menu())
    WINDOW["_DATASET_"].update(title)
    sg.Popup(f"The dataset '{title}' has been loaded successfully.\n\nSize: {DATABASE['size']} images\n", title="Dataset Selected", keep_on_top=True)

WINDOW.close()
