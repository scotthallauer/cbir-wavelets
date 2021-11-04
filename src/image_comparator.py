import numpy as np

def pass_percent(query, candidate, params):
  q, c = query, candidate
  beta = 1 - (params["percent"] / 100)
  def accept(qs, cs):
    return (qs*beta < cs and cs < qs/beta)
  return accept(q["s_c1"], c["s_c1"]) or (accept(q["s_c2"], c["s_c2"]) and accept(q["s_c3"], c["s_c3"]))

def pass_threshold(query, candidate, params):
  diff = query - candidate
  if np.sum(diff) > params["threshold"]:
    return (False, diff)
  return (True, diff)

def calc_distance(query, candidate, params, vectors=range(4), channels=range(6)):
  diffs = []
  dist = 0
  # check threshold
  for v in vectors:
    diffs.append([])
    for c in channels:
      if params["w_quad"][v] > 0 and params["w_comp"][c] > 0:
        passed, diff = pass_threshold(query[f"w_c{c+1}"][v], candidate[f"w_c{c+1}"][v], params)
        if passed:
          diffs[v].append(diff)
        else:
          return (False, -1)
      else:
        diffs[v].append(0)
  # compute distance
  for v in vectors:
    for c in channels:
      if params["w_quad"][v] > 0 and params["w_comp"][c] > 0:
        dist += params["w_quad"][v] * params["w_comp"][c] * np.linalg.norm(diffs[v][c])
  return (True, dist)

def pair2score(query, candidate, params):
  if not pass_percent(query, candidate, params):
    return (False, -1)
  return calc_distance(query, candidate, params)