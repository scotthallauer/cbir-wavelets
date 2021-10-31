import numpy as np

def pass_threshold(query_vector, match_vector, percent):
  q = query_vector
  m = match_vector
  B = 1 - (percent / 100)
  if (m['s_c1'] > q['s_c1'] * B) and (m['s_c1'] < q['s_c1'] / B):
    return True
  if (m['s_c2'] > q['s_c2'] * B) and (m['s_c2'] < q['s_c2'] / B) and (m['s_c3'] > q['s_c3'] * B) and (m['s_c3'] < q['s_c3'] / B):
    return True
  return False

def calc_distance_quick(query_vector, match_vector, params):
  q = query_vector
  m = match_vector
  dist = 0
  for i in range(3):
    dist += params["w_quad"][0] * params["w_comp"][i] * np.linalg.norm(q[f'w_c{i+1}'][0]-m[f'w_c{i+1}'][0])
  return dist

def calc_distance_full(query_vector, match_vector, params):
  q = query_vector
  m = match_vector
  dist = 0
  for i in range(4):
    for j in range(3):
      dist += params["w_quad"][i] * params["w_comp"][j] * np.linalg.norm(q[f'w_c{j+1}'][i]-m[f'w_c{j+1}'][i])
  return dist

def pair2score(query, candidate, params):
  #if not pass_threshold(query, candidate, params["percent"]):
  #  return (False, -1)
  #if calc_distance_quick(query, candidate, params) > params["threshold"]:
  #  return (False, -1)
  return (True, calc_distance_full(query, candidate, params))