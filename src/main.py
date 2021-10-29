from image_processor import img2vec, batch_resize, batch_vectorize, load_database
from image_comparator import pair2score
from shutil import copyfile
import os

root = "/Users/scott/Local/VS Code Projects/scotthallauer[cbir-wavelets]"
max_results = 20
option = 3 # 1 = resize, 2 = vectorize, 3 = query

def path(filename):
  return os.path.join(root, f"data/{filename}")

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

if option == 1:
  batch_resize(path('original'), path('resized'))

if option == 2:
  batch_vectorize(path('resized'), path('database.pickle'))

if option == 3:
  database = load_database(path("database.pickle"))
  results = search("query5.jpg", database)
  for i in range(min(max_results, len(results))):
    copyfile(path(f'original/{results[i]["file"]}'), path(f'results/result{i+1}.jpg'))