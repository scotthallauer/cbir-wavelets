import PySimpleGUI as sg
import cv2
import math
import image_processor as ip

query = {
  "image": {
    "path": "/Users/scott/Local/VS Code Projects/scotthallauer[cbir-wavelets]/data/query1.jpg",
    "tiny": None,
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

def submit_query(values):
  load_image(values["QUERY_PATH"])
  query["params"] = {
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
  window["QUERY_IMAGE"].update(data=ip.img2bytes(query["image"]["large"]))
  window["RESULTS"].update(text_color="red")

def search(query, database):
  results = []
  q = img2vec(path(query))
  for c in database['image']:
    score = pair2score(q, c['vector'])
    if score[0]:
      results.append({
        "file": c['file'],
        "score": score[1]
      })
  ordered = sorted(results, key=lambda d: d['score'])
  return ordered

def load_image(path):
  query["image"]["path"] = path
  image = cv2.imread(query["image"]["path"])
  query["image"]["tiny"] = ip.resize_image(image, (width(0.128), width(0.128)))
  query["image"]["small"] = ip.resize_image(image, (128, 128))
  query["image"]["large"] = ip.resize_image(image, (width(0.28), width(0.28)))

window_size = (1000, 600)

def width(proportion):
  return math.floor(window_size[0] * proportion)

def height(proportion):
  return math.floor(window_size[1] * proportion)

load_image(query["image"]["path"])

query_column = [
  [
    sg.Text("Query", font=("Helvetica", 20, "bold")),
  ],
  [
    sg.Image(data=ip.img2bytes(query["image"]["large"]), key="QUERY_IMAGE")
  ],
  [
    sg.Text("Path"),
    sg.Input(str(query["image"]["path"]), size=(25, 1), enable_events=True, key="QUERY_PATH"),
    sg.Button("Load")
  ],
  [
    sg.Text("percent"),
    sg.Input(str(query["params"]["percent"]), size=(25, 1), enable_events=True, key="PARAM_PERCENT"),
  ],
  [
    sg.Text("threshold"),
    sg.Input(str(query["params"]["threshold"]), size=(25, 1), enable_events=True, key="PARAM_THRESHOLD"),
  ],
  [
    sg.Text("w11"),
    sg.Input(str(query["params"]["w_quad"][0]), size=(25, 1), enable_events=True, key="PARAM_W11"),
  ],
  [
    sg.Text("w12"),
    sg.Input(str(query["params"]["w_quad"][1]), size=(25, 1), enable_events=True, key="PARAM_W12"),
  ],
  [
    sg.Text("w21"),
    sg.Input(str(query["params"]["w_quad"][2]), size=(25, 1), enable_events=True, key="PARAM_W21"),
  ],
  [
    sg.Text("w22"),
    sg.Input(str(query["params"]["w_quad"][3]), size=(25, 1), enable_events=True, key="PARAM_W22"),
  ],
  [
    sg.Text("wc1"),
    sg.Input(str(query["params"]["w_comp"][0]), size=(25, 1), enable_events=True, key="PARAM_WC1"),
  ],
  [
    sg.Text("wc2"),
    sg.Input(str(query["params"]["w_comp"][1]), size=(25, 1), enable_events=True, key="PARAM_WC2"),
  ],
  [
    sg.Text("wc3"),
    sg.Input(str(query["params"]["w_comp"][2]), size=(25, 1), enable_events=True, key="PARAM_WC3"),
  ],
  [
    sg.Text("max results"),
    sg.Slider(default_value=query["params"]["limit"], range=(1,50), orientation="h", key="PARAM_LIMIT")
  ],
  [
    sg.Button("Search")
  ]
]

result_column = [
  [
    sg.Text("Results", font=("Helvetica", 20, "bold"), key="RESULTS"),
  ]
]

for row in range(10):
  result_images = []
  for col in range(5):
    result_images.append(sg.Image(data=ip.img2bytes(query["image"]["tiny"]), visible=False, key=f"RESULT_IMAGE_{row*10 + col}"))
  result_column.append(result_images)


layout = [
  [
    sg.Column(query_column, size=(width(0.3), height(1))),
    sg.VSeparator(),
    sg.Column(result_column, size=(width(0.7), height(1)), scrollable=True, vertical_scroll_only=True),
  ]
]

window = sg.Window("CBIR Search Engine", layout)

while True:
  event, values = window.read()
  if event == "Exit" or event == sg.WIN_CLOSED:
    break
  if event == "Load":
    load_image(values["QUERY_PATH"])
    window["QUERY_IMAGE"].update(data=ip.img2bytes(query["image"]["large"]))
  if event == "Search":
    submit_query(values)

window.close()
