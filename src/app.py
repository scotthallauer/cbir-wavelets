import PySimpleGUI as sg
import cv2
import math
import image_processor as ip
from os import getcwd
from os.path import join
from dataset_manager import DatasetManager
from query_manager import QueryManager

# CONFIGURABLE PARAMETERS

WINDOW_DIM = (1200, 700)

IMAGE_DIM = (128, 128)

DEFAULT_PARAMS = {
  "percent": 50,
  "threshold": 50000,
  "w_quad": [1,1,1,1],
  "w_comp": [1,2,2,0,0,0],
  "limit": 20
}

# INTERNAL PARAMETERS

ROOT = getcwd()

WINDOW_OPENED = False

DM = DatasetManager(ROOT, IMAGE_DIM)

QM = QueryManager(ROOT, DM, IMAGE_DIM)

QUERY = {
  "image": {
    "path": join(ROOT, "src/media/query/query1.jpg")
  },
  "params": {
    "percent": DEFAULT_PARAMS["percent"],
    "threshold": DEFAULT_PARAMS["threshold"],
    "w_quad": DEFAULT_PARAMS["w_quad"],
    "w_comp": DEFAULT_PARAMS["w_comp"],
    "limit": DEFAULT_PARAMS["limit"]
  }
}

BLANK_IMAGE = ip.img2bytes(ip.resize_image(cv2.imread(join(ROOT, "src/media/blank.png")), (math.floor(WINDOW_DIM[0] * 0.137), math.floor(WINDOW_DIM[0] * 0.137))))

# FUNCTIONS

def width(proportion):
  return math.floor(WINDOW_DIM[0] * proportion)

def height(proportion):
  return math.floor(WINDOW_DIM[1] * proportion)

def update_query(values):
  QUERY["image"] = {
    "path": values["QUERY_PATH"]
  }
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
      int(values["PARAM_WC3"]),
      int(values["PARAM_WC4"]),
      int(values["PARAM_WC5"]),
      int(values["PARAM_WC6"])
    ],
    "limit": int(values["PARAM_LIMIT"])
  }
  WINDOW["QUERY_IMAGE"].update(data=ip.img2bytes(ip.resize_image(cv2.imread(QUERY["image"]["path"]), (width(0.19), width(0.19)))))

def display_results():
  for i in range(50):
    if i < min(len(QM.get_results()), QUERY["params"]["limit"]):
      WINDOW[f"RESULT_IMAGE_{i}"].update(data=ip.img2bytes(ip.resize_image(cv2.imread(join(DM.get_path(), "original", QM.get_results()[i]["file"])), (width(0.137), width(0.137)))))
      WINDOW[f"RESULT_IMAGE_{i}"].set_tooltip(f"Distance: {round(QM.get_results()[i]['score'])}")
    else:
      WINDOW[f"RESULT_IMAGE_{i}"].update(data=BLANK_IMAGE)
      WINDOW[f"RESULT_IMAGE_{i}"].set_tooltip(None)
  if len(QM.get_results()) == 0:
    WINDOW["_EXPORT_"].update(visible=False)
  else:
    WINDOW["_EXPORT_"].update(visible=True)

def clear_results():
  for i in range(50):
    WINDOW[f"RESULT_IMAGE_{i}"].update(data=BLANK_IMAGE)
  WINDOW["_EXPORT_"].update(visible=False)
  WINDOW["_STATS_"].update(visible=False)

def display_stats():
  WINDOW["_STATS_"].update(
    f"Showing {min(len(QM.get_results()), QUERY['params']['limit'])} of {len(QM.get_results())} results ({'{:.2f}'.format(QM.get_time())} seconds)", 
    visible=True)

def open_import_window():
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
    [sg.Text("To import a dataset, please provide a directory containing only\nimages (or sub-directories of images). These images will be copied\ninto the application and their feature vectors will be extracted.")],
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
      title = values["_NEW_DATASET_TITLE_"][:20]
      src = values["_NEW_DATASET_PATH_"]
      try:
        stats = DM.import_dataset(src, title)
        window.close()
        if WINDOW_OPENED:
          WINDOW["_MENU_"].update(menu_definition=generate_menu())
          WINDOW["_DATASET_"].update(f"{DM.get_title()} ({DM.database()['size']} images)")
          clear_results()
        sg.Popup(
          f"The dataset '{DM.get_title()}' has been imported successfully.\n\n" +
          f"Size: {DM.database()['size']} images\n\n" +
          f"Copy Time: {'{:.2f}'.format(stats['c'])} seconds\n" +
          f"Resize Time: {'{:.2f}'.format(stats['r'])} seconds\n" +
          f"Vectorize Time: {'{:.2f}'.format(stats['v'])} seconds\n", 
          title="Dataset Imported", 
          keep_on_top=True
        )
        break
      except Exception as e:
        print(e)
        continue

def generate_menu():
  menu_datasets = []
  for d in DM.list_datasets():
    menu_datasets.append(f"{'!' if DM.is_loaded(d['id']) else ''}{d['title']}{' (selected)' if DM.is_loaded(d['id']) else ''}::_DATASET-{d['id']}_")
  return ['&File', ['&Import Dataset...', '&Select Dataset', menu_datasets]], ['&Help', ['Online Manual']]

# INITIALISE

DM.discover_datasets()

if len(DM.list_datasets()) == 0:
  open_import_window()
else:
  DM.load_dataset(DM.list_datasets()[0]["id"])

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

CMP_WEIGHT_FRAME = [
  [sg.Slider(default_value=QUERY["params"]["w_comp"][0], range=(0,5), size=(17,15), pad=(5, (5, 0)), orientation="h", key="PARAM_WC1")],
  [sg.Text("wc1", pad=(5, (0, 5)), justification="center", expand_x=True)],
  [sg.Slider(default_value=QUERY["params"]["w_comp"][1], range=(0,5), size=(17,15), pad=(5, (5, 0)), orientation="h", key="PARAM_WC2")],
  [sg.Text("wc2", pad=(5, (0, 5)), justification="center", expand_x=True)],
  [sg.Slider(default_value=QUERY["params"]["w_comp"][2], range=(0,5), size=(17,15), pad=(5, (5, 0)), orientation="h", key="PARAM_WC3")],
  [sg.Text("wc3", pad=(5, (0, 5)), justification="center", expand_x=True)],
]

RGB_WEIGHT_FRAME = [
  [sg.Slider(default_value=QUERY["params"]["w_comp"][3], range=(0,5), size=(17,15), pad=(5, (5, 0)), orientation="h", key="PARAM_WC4")],
  [sg.Text("wr", pad=(5, (0, 5)), justification="center", expand_x=True)],
  [sg.Slider(default_value=QUERY["params"]["w_comp"][4], range=(0,5), size=(17,15), pad=(5, (5, 0)), orientation="h", key="PARAM_WC5")],
  [sg.Text("wg", pad=(5, (0, 5)), justification="center", expand_x=True)],
  [sg.Slider(default_value=QUERY["params"]["w_comp"][5], range=(0,5), size=(17,15), pad=(5, (5, 0)), orientation="h", key="PARAM_WC6")],
  [sg.Text("wb", pad=(5, (0, 5)), justification="center", expand_x=True)],
]

SELECTION_SETTING_FRAME = [
  [sg.Slider(default_value=QUERY["params"]["percent"], range=(1,100), size=(17,15), pad=(5, (5, 0)), orientation="h", key="PARAM_PERCENT")],
  [sg.Text("percent", pad=(5, (0, 5)), justification="center", expand_x=True)],
  [sg.Slider(default_value=QUERY["params"]["threshold"], range=(20000,1000000), size=(17,15), pad=(5, (5, 0)), orientation="h", key="PARAM_THRESHOLD")],
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
    sg.Text(f"{DM.get_title()} ({DM.database()['size']} images)", pad=((5, 0), 5), key="_DATASET_"),
    sg.Text("[?]", font=("Helvetica", 8), pad=((2, 5), 5), tooltip="To change dataset, go to File > Select Dataset")
  ],
  [
    sg.Column([[sg.Image(data=ip.img2bytes(ip.resize_image(cv2.imread(QUERY["image"]["path"]), (width(0.19), width(0.19)))), key="QUERY_IMAGE")]], justification='center')
  ],
  [
    sg.Text("Path"),
    sg.Input(QUERY["image"]["path"], size=(23, 1), enable_events=True, key="QUERY_PATH"),
    sg.FileBrowse(initial_folder=join(ROOT, "src/media/query"), key="_FILE_BROWSE_"),
    sg.Button("Load")
  ],
  [
    sg.Frame(title="Matrix Weights", font=("Helvetica", 11), layout=[[sg.Column(MATRIX_WEIGHT_LEFT_COLUMN), sg.Column(MATRIX_WEIGHT_RIGHT_COLUMN)]])
  ],
  [
    sg.TabGroup([
      [sg.Tab("CMP", CMP_WEIGHT_FRAME)],
      [sg.Tab("RGB", RGB_WEIGHT_FRAME)]
    ]),
    #sg.Frame(title="Channel Weights", font=("Helvetica", 11), pad=((5, 8), 5), layout=CMP_WEIGHT_FRAME),
    sg.Frame(title="Search Settings", font=("Helvetica", 11), pad=((8, 5), 5), layout=SELECTION_SETTING_FRAME),
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
    sg.Column(QUERY_COLUMN),
    sg.VSeparator(),
    sg.Column(RESULTS_COLUMN, size=(width(0.75), height(1)), scrollable=True, vertical_scroll_only=True),
  ]
]

# RUN

WINDOW = sg.Window("Wavelet Search", LAYOUT)

WINDOW_OPENED = True

while True:
  event, values = WINDOW.read()
  if event == "Exit" or event == sg.WIN_CLOSED:
    break
  if event == "Load":
    update_query(values)
  if event == "Search":
    print(f"Dataset: {DM.get_title()}")
    print(f"Query: {QUERY}")
    update_query(values)
    QM.process_query(QUERY)
    display_results()
    display_stats()
  if event == "_EXPORT_":
    filename = QM.export_results()
    filename = f"{filename.split('/')[-2]}/{filename.split('/')[-1]}"
    sg.Popup(f"Results exported to '{filename}'.", title="Export Complete", keep_on_top=True)
  if event == "Import Dataset...":
    open_import_window()
  if "_DATASET-" in event:
    idx = int(event[event.index("_DATASET-") + 9:len(event)-1])
    DM.load_dataset(idx)
    clear_results()
    WINDOW["_MENU_"].update(menu_definition=generate_menu())
    WINDOW["_DATASET_"].update(f"{DM.get_title()} ({DM.database()['size']} images)")
    sg.Popup(f"The dataset '{DM.get_title()}' has been loaded successfully.\n\nSize: {DM.database()['size']} images\n", title="Dataset Selected", keep_on_top=True)

WINDOW.close()
