"""Features and optional model-based safety shield for the official framework.

The forecast assumes stationary opponents and no newly placed opponent bombs.
Blast rays pass through crates; bombs do not chain-react in this framework.
AI-assisted implementation, pending student review and refinement.
"""
from collections import deque
from functools import lru_cache

import numpy as np
import settings as settings

ACTIONS = ('UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB')
DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))


def blast(field, position):
    cells = [position]
    for dx, dy in DELTAS:
        for radius in range(1, settings.BOMB_POWER + 1):
            p = (position[0] + dx * radius, position[1] + dy * radius)
            if not (0 <= p[0] < field.shape[0] and 0 <= p[1] < field.shape[1]):
                break
            if field[p] == -1:
                break
            cells.append(p)
    return cells


@lru_cache(maxsize=256)
def crate_yields(shape, data):
    field = np.frombuffer(data, dtype=np.int8).reshape(shape)
    result = np.zeros(shape, dtype=np.int8)
    for x, y in zip(*np.where(field == 0)):
        result[x, y] = sum(field[p] == 1 for p in blast(field, (x, y)))
    return result


def forecast(state, new_bomb=False):
    """Cells from which survival is possible after the next action.

    t=1 denotes the end of the next action. Existing timer k explodes at
    t=k+1; a newly dropped timer-4 bomb explodes at t=5. Crates are removed
    after movement; a tile opens for movement one step after its destruction.
    """
    field = state['field']
    bombs = list(state['bombs'])
    if new_bomb:
        bombs.append((state['self'][3], settings.BOMB_TIMER))
    horizon = max([int(state['explosion_map'].max()), 1]
                  + [timer + settings.EXPLOSION_TIMER for _, timer in bombs])
    danger = np.zeros((horizon + 1,) + field.shape, dtype=bool)
    opens = np.where(field == 0, 0, horizon + 2)
    occupied_until = np.zeros(field.shape, dtype=int)
    for position, timer in bombs:
        detonation = timer + 1
        occupied_until[tuple(position)] = detonation
        for p in blast(field, position):
            danger[detonation:detonation + settings.EXPLOSION_TIMER, p[0], p[1]] = True
            if field[p] == 1:
                opens[p] = min(opens[p], detonation + 1)
    for t in range(1, horizon + 1):
        danger[t] |= state['explosion_map'] >= t
    for other in state['others']:
        opens[tuple(other[3])] = horizon + 2
    viable = None
    for t in range(horizon, 0, -1):
        standing = (opens <= t) & ~danger[t]
        if viable is None:
            viable = standing
            continue
        destinations = viable & (occupied_until < t + 1) & (opens <= t + 1)
        following = viable.copy()  # WAIT may stay on the bomb underneath self.
        following[:, 1:] |= destinations[:, :-1]
        following[:-1, :] |= destinations[1:, :]
        following[:, :-1] |= destinations[:, 1:]
        following[1:, :] |= destinations[:-1, :]
        viable = standing & following
    return viable


def distance_map(free, targets):
    distances = np.full(free.shape, -1, dtype=np.int16)
    queue = deque()
    for p in targets:
        p = tuple(p)
        if free[p]:
            distances[p] = 0
            queue.append(p)
    while queue:
        x, y = queue.popleft()
        for dx, dy in DELTAS:
            p = (x + dx, y + dy)
            if 0 <= p[0] < free.shape[0] and 0 <= p[1] < free.shape[1] and free[p] and distances[p] < 0:
                distances[p] = distances[x, y] + 1
                queue.append(p)
    return distances


def describe(state, anticipate=False, attack_features=False):
    field = state['field']
    position = tuple(state['self'][3])
    free = field == 0
    for p, _ in state['bombs']:
        free[tuple(p)] = False
    for other in state['others']:
        free[tuple(other[3])] = False
    neighbors = [(position[0] + dx, position[1] + dy) for dx, dy in DELTAS]
    legal = np.array([free[p] for p in neighbors] + [True, bool(state['self'][2])])
    viable = forecast(state)
    safe = np.array([viable[p] for p in neighbors] + [viable[position], False]) & legal
    if legal[5]:
        safe[5] = forecast(state, new_bomb=True)[position]

    robust_safe = safe.copy()
    if anticipate:
        bomb_positions = {tuple(p) for p, _ in state['bombs']}
        hypothetical = [(tuple(other[3]), settings.BOMB_TIMER) for other in state['others']
                        if other[2] and tuple(other[3]) not in bomb_positions]
        if hypothetical:
            predicted = {**state, 'bombs': list(state['bombs']) + hypothetical}
            robust_viable = forecast(predicted)
            robust_safe[:5] &= np.array([robust_viable[p] for p in neighbors] + [robust_viable[position]])
            if legal[5]:
                robust_safe[5] &= forecast(predicted, new_bomb=True)[position]

    # Start is included for BFS even while standing on one's own bomb.
    free[position] = True
    yields = crate_yields(field.shape, field.astype(np.int8).tobytes())
    distances = distance_map(free, state['coins'])
    target_type = 1
    enemy_positions = [tuple(other[3]) for other in state['others']]
    # A bomb can threaten an enemy from any unobstructed blast-ray tile.
    attack_targets = {p for enemy in enemy_positions for p in blast(field, enemy)
                      if free[p]}
    attack_distances = distance_map(free, attack_targets)
    attack_distance = int(attack_distances[position])
    coin_distance = int(distances[position])
    # Nearby attack opportunities take priority over distant resources.
    if 0 <= attack_distance <= 3 and (coin_distance < 0 or coin_distance > 2):
        distances = attack_distances
        target_type = 3
    if distances[position] < 0:
        targets = list(zip(*np.where((yields > 0) & free)))
        distances = distance_map(free, targets)
        target_type = 2 if distances[position] >= 0 else 0
        if target_type == 0 and attack_distance >= 0:
            distances = attack_distances
            target_type = 3
    distance = int(distances[position])
    directions = tuple(int(0 <= distances[p] < distance) for p in neighbors)
    progress = tuple(int(np.sign(distance - distances[p])) if distances[p] >= 0 else -1
                     for p in neighbors)
    codes = tuple(int(legal[i]) + int(safe[i]) for i in range(6))
    threatened = bool(state['explosion_map'][position] > 0 or any(
        position in blast(field, tuple(p)) for p, _ in state['bombs']))
    category = 0 if distance <= 0 else (1 if distance == 1 else 2)
    enemy_hits = sum(enemy in blast(field, position) for enemy in enemy_positions)
    enemy_bomb_ready = any(other[2] and abs(other[3][0] - position[0]) + abs(other[3][1] - position[1]) <= 4
                           for other in state['others'])
    features = codes + directions + (target_type, min(int(yields[position]), 3), int(threatened), category,
                                      min(enemy_hits, 2), int(enemy_bomb_ready))
    if attack_features:
        features += attack_context(state, safe[5])
    return {'key': features, 'legal': legal, 'safe': safe, 'robust_safe': robust_safe,
            'distance': max(distance, 0), 'target': target_type,
            'yield': int(yields[position]), 'threatened': threatened, 'progress': progress,
            'enemy_hits': enemy_hits}


def attack_context(state, safe_bomb):
    """Two coarse descriptors, not a prediction of an opponent's policy.

    Enemy exits: fewest currently legal moves among enemies on our blast ray,
    capped at two; three means no threatened opponent. Escape margin: static
    shortest route outside our prospective blast, compared to four remaining
    moves before detonation. The original timed shield still decides safety.
    """
    field = state['field']
    position = tuple(state['self'][3])
    ray = set(blast(field, position))
    blocked = {tuple(p) for p, _ in state['bombs']}
    blocked |= {tuple(a[3]) for a in state['others']} | {position}
    exits = []
    for other in state['others']:
        x, y = other[3]
        if (x, y) in ray:
            exits.append(sum(field[x + dx, y + dy] == 0 and
                             (x + dx, y + dy) not in blocked for dx, dy in DELTAS))
    enemy_code = min(min(exits), 2) if exits else 3
    if not state['self'][2]:
        return enemy_code, 3
    if not safe_bomb:
        return enemy_code, 0
    free = field == 0
    for p in blocked:
        free[p] = False
    free[position] = True
    targets = [tuple(p) for p in zip(*np.where(free)) if tuple(p) not in ray]
    distance = int(distance_map(free, targets)[position])
    # Unreachable static paths may become possible after crate destruction;
    # treat them as tight rather than overriding the timed safety forecast.
    margin = settings.BOMB_TIMER - distance if distance >= 0 else 0
    return enemy_code, 2 if margin >= 2 else 1


def prior_key(key):
    """Map fighting features to the previously learned resource policy."""
    base = list(key[:14])
    if base[10] == 3:
        base[10] = 2
        base[11] = max(base[11], key[14])
    return tuple(base)


def allowed(info, shield):
    if shield and info.get('robust_safe', np.array([], dtype=bool)).any():
        return info['robust_safe']
    return info['safe'] if shield and info['safe'].any() else info['legal']
