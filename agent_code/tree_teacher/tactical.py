"""Public geometric attack/resource context, without predicting hidden actions."""
import numpy as np
import settings
from .features import DELTAS, attack_context, blast, describe, distance_map


def enemy_exit_counts(state, origin):
    field = state['field']
    ray = set(blast(field, origin))
    blocked = {tuple(p) for p,_ in state['bombs']} | {tuple(o[3]) for o in state['others']} | {origin}
    return [sum(field[o[3][0]+dx,o[3][1]+dy] == 0
                and (o[3][0]+dx,o[3][1]+dy) not in blocked for dx,dy in DELTAS)
            for o in state['others'] if tuple(o[3]) in ray]


def tactical_key(state, info=None):
    info = describe(state) if info is None else info
    position = tuple(state['self'][3])
    field = state['field']
    blocked = {tuple(p) for p,_ in state['bombs']} | {tuple(o[3]) for o in state['others']}
    current = attack_context(state, info['safe'][5])
    neighboring = []
    for dx,dy in DELTAS:
        dest = (position[0]+dx,position[1]+dy)
        if field[dest] != 0 or dest in blocked:
            neighboring.append(4)  # Unavailable candidate position.
        else:
            exits = enemy_exit_counts(state,dest)
            neighboring.append(min(min(exits),2) if exits else 3)
    free = field == 0
    for p in blocked: free[p] = False
    free[position] = True
    distances = distance_map(free,state['coins'])
    distance = int(distances[position])
    coin_directions = [int(0 <= distances[position[0]+dx,position[1]+dy] < distance)
                       for dx,dy in DELTAS]
    coin_distance = 0 if distance < 0 else 1 if distance <= 1 else 2 if distance <= 4 else 3
    remaining = max(0,settings.MAX_STEPS-state['step']+1)
    time_bucket = 0 if remaining <= 10 else 1 if remaining <= 40 else 2
    key = tuple(info['key'][:16])+tuple(current)+tuple(neighboring)+tuple(coin_directions)+(coin_distance,time_bucket)
    assert len(key) == 28
    return np.asarray(key,dtype=np.int8)


def open_attack_bonus(state, action, info=None):
    """Existing 1.5-per-ray-enemy bonus attributable to >=2 geometric exits.

    This is a shaping component, not a calibrated kill probability. Preserve
    event rewards and crate/coin shaping; this value is only subtracted by the
    explicitly named reward-ablation candidate.
    """
    info = describe(state) if info is None else info
    if action != 'BOMB' or not info['safe'][5]:
        return 0.
    return 1.5*sum(n >= 2 for n in enemy_exit_counts(state,tuple(state['self'][3])))
