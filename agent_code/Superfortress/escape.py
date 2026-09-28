"""Risk preference among shield-approved actions; preserves learned state keys."""
import numpy as np

from .features import DELTAS, blast, distance_map


def effective_bomb_cost(state, base_cost, mode='full', highest_score=None):
    if mode == 'off':
        return 0.
    if mode == 'late_behind':
        leader = max([state['self'][1]] + [o[1] for o in state['others']])
        if highest_score is not None:
            leader = max(leader, highest_score)
        if not state['coins'] and not np.any(state['field'] == 1) and state['self'][1] < leader:
            return 0.
    elif mode != 'full':
        raise ValueError(f'Unknown bomb cost mode: {mode}')
    return base_cost


def escape_costs(state, info, bomb_cost=1., contest_scale=1.):
    field = state['field']
    position = tuple(state['self'][3])
    free = field == 0
    for p, _ in state['bombs']:
        free[tuple(p)] = False
    for other in state['others']:
        free[tuple(other[3])] = False
    free[position] = True
    # Count opponents that can contest a destination this turn. A risk, not
    # a hard obstacle: permanently blocking every such tile can trap us too.
    contest = np.zeros(field.shape)
    for other in state['others']:
        x, y = other[3]
        for dx, dy in DELTAS:
            p = x + dx, y + dy
            if free[p] and p != position:
                contest[p] += 1
    danger = state['explosion_map'] > 0
    for p, _ in state['bombs']:
        for cell in blast(field, p):
            danger[cell] = True
    costs = np.zeros(6)
    destinations = [(position[0] + dx, position[1] + dy) for dx, dy in DELTAS] + [position, position]
    if info['threatened']:
        targets = list(zip(*np.where(free & ~danger)))
        distance = distance_map(free, targets)
        for action, p in enumerate(destinations):
            if info['legal'][action]:
                # Prefer reaching a tile outside all currently known blasts
                # sooner. The exact timed shield still decides safety.
                costs[action] += .8 * (int(distance[p]) if distance[p] >= 0 else 8)
    for action, p in enumerate(destinations[:4]):
        costs[action] += contest_scale * (4. if info['threatened'] else 1.) * contest[p]
    # Bomb placement consumes an escape turn. Near opponents, ask the learned
    # attack value to justify the added risk rather than forbidding bombing.
    if any(abs(position[0] - o[3][0]) + abs(position[1] - o[3][1]) <= 4 for o in state['others']):
        costs[5] += bomb_cost
    return costs
