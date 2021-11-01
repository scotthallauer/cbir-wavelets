import numpy as np

def pass_percent(query, candidate, params):
  q, c = query, candidate
  beta = 1 - (params["percent"] / 100)
  def accept(qs, cs):
    return (qs*beta < cs and cs < qs/beta)
  return accept(q["s_c1"], c["s_c1"]) or (accept(q["s_c2"], c["s_c2"]) and accept(q["s_c3"], c["s_c3"]))

def calc_distance(query, candidate, params, vectors=range(4), channels=range(6)):
  q, c = query, candidate
  dist = 0
  for i in vectors:
    for j in channels:
      if params["w_quad"][i] > 0 and params["w_comp"][j] > 0:
        dist += params["w_quad"][i] * params["w_comp"][j] * np.linalg.norm(q[f'w_c{j+1}'][i]-c[f'w_c{j+1}'][i])
  return dist

def pair2score(query, candidate, params):
  if not pass_percent(query, candidate, params):
    return (False, -1)
  d1 = calc_distance(query, candidate, params, vectors=[0])
  if d1 > params["threshold"]:
    return (False, -1)
  return (True, d1 + calc_distance(query, candidate, params, vectors=[1, 2, 3]))