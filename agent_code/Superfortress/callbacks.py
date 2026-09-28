"""Adopted E policy: frozen block Q and both resource guards for classic."""
import json
import os
from pathlib import Path

import numpy as np

from . import base
from .features import ACTIONS, describe
from .history import PursuitLedger
from .resource_inventory import CoinInventory
from .switching import SwitchController

MODES = {'A': (False, False, False), 'B': (True, False, False),
         'C': (True, True, False), 'D': (True, False, True), 'E': (True, True, True)}


def encode(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def setup(self):
    base.setup(self)
    self.guard_mode = 'E'
    if self.guard_mode not in MODES:
        raise ValueError('QL_GUARD_MODE must be A, B, C, D, or E')
    self.guard_seed = 42
    # Public classic tournament rule, fixed with the accepted policy. This
    # must work without the evaluation harness providing environment values.
    self.coin_total = 9
    self.guard_round = None
    trace = ''  # Submission inference never opens experiment trace files.
    self.guard_trace = Path(trace).open('x', encoding='utf-8', buffering=1) if trace else None


def act(self, game_state):
    enabled, inventory_guard, bomb_guard = MODES[self.guard_mode]
    if self.guard_round != game_state['round']:
        self.guard_round = game_state['round']
        self.rng = np.random.default_rng(np.random.SeedSequence([self.guard_seed, self.guard_round, 101]))
        self.proposal_rng = np.random.default_rng(np.random.SeedSequence([self.guard_seed, self.guard_round, 102]))
        self.guard_ledger = PursuitLedger()
        self.guard_controller = SwitchController(inventory_guard, bomb_guard)
        self.coin_inventory = CoinInventory(self.coin_total)
    inventory = self.coin_inventory.observe(game_state)
    ledger = self.guard_ledger
    history = ledger.observe(game_state)
    mask, scores, urgent = base.scored_actions(self, game_state)
    info = describe(game_state)
    candidates = np.flatnonzero(mask)
    best = candidates[np.isclose(scores[candidates], scores[candidates].max())]
    baseline = int(self.rng.choice(best))
    selected = baseline
    details = dict(activated=False, reason='off', commitment=None, allowed=[], guard=None, accepted=False)
    if enabled:
        options, details = self.guard_controller.select_guarded(
            game_state, info, mask, scores, ledger, history, self.proposal_rng, inventory, baseline)
        if options and baseline not in options:
            selected = int(self.proposal_rng.choice(options))
        self.guard_controller.after_action(selected, details)
    assert mask[selected]
    assert not info['threatened'] or selected == baseline
    assert not bomb_guard or selected != 5 or baseline == 5
    ledger.remember_action(game_state, ACTIONS[selected])
    if self.guard_trace:
        record = dict(round=game_state['round'], step=game_state['step'], mode=self.guard_mode,
                      position=list(game_state['self'][3]), own_score=game_state['self'][1],
                      history=history, original_target=info['target'], threatened=bool(info['threatened']),
                      urgent_corrected=urgent, candidate_mask=mask.tolist(), scores=scores.tolist(),
                      baseline_best=best.tolist(), baseline_action=ACTIONS[baseline], action=ACTIONS[selected],
                      changed=selected != baseline, inventory=inventory, decision=details,
                      random_streams=[self.guard_seed, self.guard_round, 101, 102])
        self.guard_trace.write(json.dumps(record, default=encode) + '\n')
    return ACTIONS[selected]
