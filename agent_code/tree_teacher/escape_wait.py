"""A scale-free, inference-only correction for delaying an available retreat."""
import numpy as np
from .features import DELTAS, blast, distance_map


def retreat_options(state, info, mask, values):
    candidates = np.flatnonzero(mask)
    best = candidates[np.isclose(values[candidates], values[candidates].max())]
    if not info['threatened'] or best.tolist() != [4] or not info['safe'].any():
        return [], None
    pos = tuple(state['self'][3]); field = state['field']
    free = field == 0
    for p, _ in state['bombs']: free[tuple(p)] = False
    for other in state['others']: free[tuple(other[3])] = False
    free[pos] = True
    danger = state['explosion_map'] > 0
    for p, _ in state['bombs']:
        for cell in blast(field, tuple(p)): danger[cell] = True
    distance = distance_map(free, list(zip(*np.where(free & ~danger))))
    current = int(distance[pos])
    moves = [(pos[0]+dx,pos[1]+dy) for dx,dy in DELTAS]
    distances = [int(distance[p]) for p in moves]+[current,current]
    options = [a for a in range(4) if mask[a] and info['safe'][a]
               and 0 <= distances[a] < current]
    if not options: return [], distances
    shortest = min(distances[a] for a in options)
    options = [a for a in options if distances[a] == shortest]
    maximum = max(values[a] for a in options)
    return [a for a in options if np.isclose(values[a], maximum)], distances
