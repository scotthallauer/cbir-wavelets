import dataset_processor as dp
from os.path import join

root = "/Users/scott/Local/VS Code Projects/scotthallauer[cbir-wavelets]"
dim = (128, 128)
option = 2 # 1 = resize, 2 = vectorize, 3 = query

if option == 1:
  dp.batch_resize(join(root, "data/original"), join(root, "data/resized"), dim)

if option == 2:
  dp.batch_vectorize(join(root, "data/resized"), join(root, "data/database.pickle"), dim)

if option == 3:
  db = dp.load_database(join(root, "data/database.pickle"))
  print(db["size"])