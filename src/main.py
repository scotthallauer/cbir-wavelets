from image_processor import img2vec, batch_resize, batch_vectorize, load_database
from image_comparator import pair2score
from shutil import copyfile
import os

root = "/Users/scott/Local/VS Code Projects/scotthallauer[cbir-wavelets]"
max_results = 20
option = 3 # 1 = resize, 2 = vectorize, 3 = query

def path(filename):
  return os.path.join(root, f"data/{filename}")

def search(query, database, params):
  results = []
  q = img2vec(path(query))
  for c in database['image']:
    score = pair2score(q, c['vector'], params)
    if score[0]:
      results.append({
        "file": c['file'],
        "score": score[1]
      })
  ordered = sorted(results, key=lambda d: d['score'])
  return ordered

if option == 1:
  batch_resize(path('original'), path('resized'))

if option == 2:
  batch_vectorize(path('resized'), path('database.pickle'))

if option == 3:
  database = load_database(path("database.pickle"))
  params = {
    "percent": 50,
    "threshold": 100000,
    "w_quad": [1,1,1,1],
    "w_comp": [1,2,2],
    "limit": 20
  }
  results = search("query5.jpg", database, params)
  for i in range(min(params["limit"], len(results))):
    copyfile(path(f'resized/{results[i]["file"]}'), path(f'results/result{i+1}.jpg'))