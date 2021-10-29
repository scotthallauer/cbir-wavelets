import numpy as np

percent = 50
distance_threshold = 100000
w11, w12, w21, w22 = 3, 1, 1, 1
wc = [1, 2, 2]

def pass_threshold(query_vector, match_vector):
  q = query_vector
  m = match_vector
  B = 1 - (percent / 100)
  if (m['s_c1'] > q['s_c1'] * B) and (m['s_c1'] < q['s_c1'] / B):
    return True
  if (m['s_c2'] > q['s_c2'] * B) and (m['s_c2'] < q['s_c2'] / B) and (m['s_c3'] > q['s_c3'] * B) and (m['s_c3'] < q['s_c3'] / B):
    return True
  return False

def calc_distance_quick(query_vector, match_vector):
  q = query_vector
  m = match_vector
  dist = 0
  for i in range(3):
    dist += w11 * wc[i] * np.linalg.norm(q[f'w_c{i+1}'][0]-m[f'w_c{i+1}'][0])
  return dist

def calc_distance_full(query_vector, match_vector):
  q = query_vector
  m = match_vector
  t1 = 0
  for i in range(3):
    t1 += w11 * wc[i] * np.linalg.norm(q[f'w_c{i+1}'][0]-m[f'w_c{i+1}'][0])
  t2 = 0
  for i in range(3):
    t2 += w12 * wc[i] * np.linalg.norm(q[f'w_c{i+1}'][1][0]-m[f'w_c{i+1}'][1][0])
  t3 = 0
  for i in range(3):
    t3 += w21 * wc[i] * np.linalg.norm(q[f'w_c{i+1}'][1][1]-m[f'w_c{i+1}'][1][1])
  t4 = 0
  for i in range(3):
    t4 += w22 * wc[i] * np.linalg.norm(q[f'w_c{i+1}'][1][2]-m[f'w_c{i+1}'][1][2])
  return t1 + t2 + t3 + t4

def pair2score(query, candidate):
  if not pass_threshold(query, candidate):
    return (False, -1)
  if calc_distance_quick(query, candidate) > distance_threshold:
    return (False, -1)
  return (True, calc_distance_full(query, candidate))