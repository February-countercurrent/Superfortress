"""Public-observation pursuit history, derived from the validated stall audit."""
from collections import Counter
from dataclasses import dataclass

import numpy as np
import settings
from .features import ACTIONS, DELTAS, blast, crate_yields, distance_map


@dataclass
class Cycle:
    placed: int
    position: tuple
    ray: set
    targets: set
    epoch: int
    retreated: bool = False


class PursuitLedger:
    def __init__(self):
        self.round = None
        self.step = 0
        self.score = 0
        self.crates = set()
        self.coins = set()
        self.epoch = 0
        self.last_score = 0
        self.last_progress = 0
        self.proposal = None
        self.pending = None
        self.streaks = {}
        self.counters = Counter()
        self.completed = []
        self._remembered = False

    def observe(self, state):
        if state['round'] != self.round:
            self.__init__()
            self.round = state['round']
        step = int(state['step'])
        if step != self.step + 1:
            raise ValueError('Expected contiguous pre-action observations starting at step 1')
        if self.step and not self._remembered:
            raise ValueError('Previous own action is missing')
        score = int(state['self'][1])
        crates = set(map(tuple, np.argwhere(state['field'] == 1)))
        coins = set(map(tuple, state['coins']))
        alive = {other[0] for other in state['others']}
        self.streaks = {name: count for name, count in self.streaks.items() if name in alive}
        if self.step:
            assert score >= self.score
            gains = score > self.score
            opened = bool(self.crates - crates)
            revealed = bool(coins - self.coins)
            if gains:
                self.last_score = step - 1
            if gains or opened or revealed:
                self.epoch += 1
                self.last_progress = step - 1
                self.streaks.clear()
                self.counters['progress_resets'] += 1
            self.counters['score_progress'] += int(gains)
            self.counters['map_opening_progress'] += int(opened)
            self.counters['coin_reveal_progress'] += int(revealed)
        if self.proposal is not None:
            cycle = self.proposal
            confirmed = (not state['self'][2] and
                         any(tuple(p) == cycle.position and timer == settings.BOMB_TIMER - 1
                             for p, timer in state['bombs']))
            if confirmed:
                assert self.pending is None, 'Own unresolved bomb overlap'
                self.pending = cycle
                self.counters['confirmed_bombs'] += 1
                self.counters['confirmed_ray_bombs'] += int(bool(cycle.targets))
                if not cycle.targets:
                    self.streaks.clear()
            else:
                self.counters['unconfirmed_bomb_actions'] += 1
            self.proposal = None
        completed = None
        if self.pending is not None:
            cycle = self.pending
            cycle.retreated |= tuple(state['self'][3]) not in cycle.ray
            position = cycle.position
            ready = (step > cycle.placed + settings.BOMB_TIMER + settings.EXPLOSION_TIMER
                     and state['self'][2])
            if ready:
                empty = cycle.epoch == self.epoch
                targets = cycle.targets & alive
                countable = bool(targets) and empty and cycle.retreated
                if countable:
                    self.streaks = {name: self.streaks.get(name, 0) + 1 for name in targets}
                else:
                    self.streaks.clear()
                completed = dict(placed=cycle.placed, observed_complete=step,
                                 targets=sorted(cycle.targets), alive_targets=sorted(targets),
                                 no_observed_progress=empty, retreated=cycle.retreated,
                                 countable=countable, streak=max(self.streaks.values(), default=0))
                self.completed.append(completed)
                self.counters['completed_bombs'] += 1
                self.counters['completed_ray_bombs'] += int(bool(cycle.targets))
                self.counters['empty_completed_ray_bombs'] += int(countable)
                self.pending = None
        self.score, self.crates, self.coins = score, crates, coins
        self.step = step
        self._remembered = False
        return dict(streak=max(self.streaks.values(), default=0),
                    targets=sorted(self.streaks), scoreless_actions=step - 1 - self.last_score,
                    no_progress_actions=step - 1 - self.last_progress,
                    pending_cycle=self.pending is not None, completed=completed)

    def remember_action(self, state, action):
        if self._remembered or state['round'] != self.round or state['step'] != self.step:
            raise ValueError('Observe once, then remember exactly one own action')
        self._remembered = True
        if action != 'BOMB' or not state['self'][2]:
            return
        position = tuple(state['self'][3])
        ray = set(blast(state['field'], position))
        targets = {other[0] for other in state['others'] if tuple(other[3]) in ray}
        self.proposal = Cycle(self.step, position, ray, targets, self.epoch)


def resource_options(state, mask):
    """Snapshot resource approaches; no estimated collection/kill probability."""
    position = tuple(state['self'][3])
    field = state['field']
    free = field == 0
    for p, _ in state['bombs']:
        free[tuple(p)] = False
    for other in state['others']:
        free[tuple(other[3])] = False
    free[position] = True
    remaining = settings.MAX_STEPS - state['step'] + 1
    yields = crate_yields(field.shape, field.astype(np.int8).tobytes())

    def approach(targets, is_crate):
        distances = distance_map(free, targets)
        distance = int(distances[position])
        actions = []
        enough_time = distance >= 0 and distance + (settings.BOMB_TIMER + 1 if is_crate else 0) <= remaining
        if enough_time:
            for i, (dx, dy) in enumerate(DELTAS):
                p = position[0] + dx, position[1] + dy
                if mask[i] and free[p] and 0 <= distances[p] < distance:
                    actions.append(ACTIONS[i])
            if is_crate and distance == 0 and mask[5] and state['self'][2]:
                actions.append('BOMB')
        return dict(distance=distance, reachable=distance >= 0,
                    enough_time=enough_time, actions=actions)

    coin = approach(state['coins'], False)
    crate = approach(list(map(tuple, np.argwhere((yields > 0) & free))), True)
    preferred = 'coin' if coin['actions'] else 'crate' if crate['actions'] else None
    return dict(coin=coin, crate=crate, preferred=preferred,
                actions=(coin if preferred == 'coin' else crate)['actions'] if preferred else [],
                visible_coins=len(state['coins']), crates=int((field == 1).sum()))


def eligible(history, info, threshold=2):
    return history['streak'] >= threshold and not info['threatened'] and info['target'] == 3
