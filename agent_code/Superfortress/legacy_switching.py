"""A fixed, bounded resource commitment after repeated unproductive attacks."""
import numpy as np
import settings

from .features import ACTIONS, DELTAS, blast, crate_yields, distance_map

MAX_COMMITMENT = 8
REQUIRED_CYCLES = 2


def walkable(state):
    free = state['field'] == 0
    for p, _ in state['bombs']:
        free[tuple(p)] = False
    for enemy in state['others']:
        free[tuple(enemy[3])] = False
    free[tuple(state['self'][3])] = True
    return free


def nearest_attackers(state, free):
    position = tuple(state['self'][3])
    distances = distance_map(free, [position])
    by_enemy = {}
    for enemy in state['others']:
        ds = [int(distances[p]) for p in blast(state['field'], tuple(enemy[3]))
              if free[p] and distances[p] >= 0]
        if ds:
            by_enemy[enemy[0]] = min(ds)
    best = min(by_enemy.values(), default=-1)
    return sorted(name for name, d in by_enemy.items() if d == best)


def toward(state, free, target, kind, mask):
    p = tuple(state['self'][3])
    distances = distance_map(free, [target])
    d = int(distances[p])
    choices = []
    if d > 0:
        for i, (dx, dy) in enumerate(DELTAS):
            q = p[0] + dx, p[1] + dy
            if mask[i] and free[q] and 0 <= distances[q] < d:
                choices.append(i)
    elif d == 0 and kind == 'crate' and state['self'][2] and mask[5]:
        choices = [5]
    return d, choices


def resource_goals(state, mask, scores, free):
    """Nearest feasible goals, then original Q ranking; exact ties are random."""
    p = tuple(state['self'][3])
    from_self = distance_map(free, [p])
    yields = crate_yields(state['field'].shape, state['field'].astype(np.int8).tobytes())
    sources = [('coin', list(map(tuple, state['coins']))),
               ('crate', list(map(tuple, np.argwhere((yields > 0) & free))))]
    remaining = settings.MAX_STEPS - state['step'] + 1
    for kind, targets in sources:
        candidates = []
        for target in targets:
            d = int(from_self[target])
            action_cost = d + int(kind == 'crate')
            time_cost = d + (settings.BOMB_TIMER + 1 if kind == 'crate' else 0)
            if d < 0 or action_cost > MAX_COMMITMENT or time_cost > remaining:
                continue
            d, allowed = toward(state, free, target, kind, mask)
            if allowed:
                candidates.append(dict(kind=kind, target=target, distance=d,
                                       value=float(np.max(scores[allowed]))))
        if candidates:
            distance = min(c['distance'] for c in candidates)
            candidates = [c for c in candidates if c['distance'] == distance]
            value = max(c['value'] for c in candidates)
            return sorted([c for c in candidates if np.isclose(c['value'], value)],
                          key=lambda c: c['target'])
    return []


class SwitchController:
    def __init__(self):
        self.active = None
        self.next_id = 1

    def select(self, state, info, mask, scores, ledger, history, rng):
        """Return constrained choices without changing the existing Q inputs."""
        details = dict(activated=False, reason='base', commitment=None, allowed=[])
        if self.active is not None:
            a = self.active
            reason = None
            if info['threatened']:
                reason = 'danger'
            elif ledger.epoch != a['epoch']:
                reason = 'progress'
            elif state['step'] - a['started'] >= MAX_COMMITMENT:
                reason = 'timeout'
            elif not set(a['enemies']) & {o[0] for o in state['others']}:
                reason = 'enemy_disappeared'
            elif a['kind'] == 'coin' and a['target'] not in set(map(tuple, state['coins'])):
                reason = 'target_disappeared'
            elif a['kind'] == 'crate' and not any(state['field'][p] == 1 for p in blast(state['field'], a['target'])):
                reason = 'target_disappeared'
            if reason:
                details.update(reason=reason, ended=a.copy())
                self.active = None
                return [], details
        if self.active is None:
            if info['threatened'] or info['target'] != 3 or history['streak'] < REQUIRED_CYCLES or ledger.pending is not None:
                return [], details
            free = walkable(state)
            enemies = [name for name in nearest_attackers(state, free)
                       if ledger.streaks.get(name, 0) >= REQUIRED_CYCLES]
            if not enemies:
                details['reason'] = 'different_attack_target'
                return [], details
            goals = resource_goals(state, mask, scores, free)
            if not goals:
                details['reason'] = 'no_short_resource_goal'
                return [], details
            goal = goals[int(rng.choice(len(goals)))] if len(goals) > 1 else goals[0]
            self.active = dict(id=self.next_id, kind=goal['kind'], target=goal['target'],
                               started=state['step'], epoch=ledger.epoch, enemies=enemies,
                               initial_distance=goal['distance'])
            self.next_id += 1
            # Consume this evidence once. A new attempt needs two fresh cycles.
            ledger.streaks.clear()
            details['activated'] = True
        a = self.active
        d, choices = toward(state, walkable(state), a['target'], a['kind'], mask)
        if not choices:
            details.update(reason='no_safe_progress', ended=a.copy())
            self.active = None
            return [], details
        scores_here = scores[choices]
        best = [choices[i] for i in np.flatnonzero(np.isclose(scores_here, scores_here.max()))]
        details.update(reason='resource_commitment', commitment=a.copy(), allowed=best, distance=d)
        return best, details

    def after_action(self, index, details):
        if self.active is not None and details['reason'] == 'resource_commitment' and index == 5:
            details['ended'] = self.active.copy()
            details['end_after_action'] = 'resource_bomb'
            self.active = None
