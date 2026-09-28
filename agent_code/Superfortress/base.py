"""Symmetry-averaged block Q with a narrow urgent-escape correction."""
import json
import os
from pathlib import Path
import numpy as np
from .features import ACTIONS, allowed, describe
from .escape import escape_costs
from .dynamic import candidate_mask, candidate_mask_with_status
from .symmetry_maps import PERMUTATIONS, transform_key

# Use exactly the original 16 observations. Action-specific coefficients
# preserve interactions within each block; no new tactical observations yet.
BLOCKS = ((), (0, 1, 2, 3, 4, 5, 12), (6, 7, 8, 9, 10, 13),
          (10, 11, 13), (10, 14, 15), (5, 11, 12, 14), (10, 12, 14, 15))


def active_tiles(key):
    if len(key) != 16:
        raise ValueError('Expected original 16-feature observation')
    return [(i, *[key[j] for j in group]) for i, group in enumerate(BLOCKS)]


def values_for(self, key):
    return sum((self.q.get(tile, np.zeros(6)) for tile in active_tiles(key)), np.zeros(6))


def averaged_values(self, key):
    # p maps original actions to transformed actions. Index by p to align each
    # prediction back to the actual board before averaging. Keep all eight
    # group elements: identical states can still permute directional actions.
    return sum((values_for(self, transform_key(key,p))[list(p)]
                for p in PERMUTATIONS),np.zeros(6))/len(PERMUTATIONS)


def setup(self):
    if self.train:
        raise ValueError('q_guard submission is inference-only')
    self.rng = np.random.default_rng(42)
    self.model_path = Path(__file__).resolve().parent / 'model.json'
    data = json.loads(self.model_path.read_text(encoding='utf-8'))
    if data['version'] != 5 or data['actions'] != list(ACTIONS) or data['blocks'] != [list(b) for b in BLOCKS]:
        raise ValueError('Incompatible linear Q checkpoint')
    self.q = {tuple(map(int, k.split(','))): np.array(v, dtype=float) for k, v in data['q'].items()}
    self.visits = {tuple(map(int, k.split(','))): np.array(v, dtype=int) for k, v in data.get('visits', {}).items()}
    self.episodes = data['episodes']
    self.prior = {}
    self.training_config = data.get('config', {})
    self.epsilon = 0.
    self.shield, self.anticipate, self.attack_features = True, False, False


def action_mask(game_state, info):
    return candidate_mask(game_state, info, allowed(info, True))


def scored_actions(self, game_state):
    info = describe(game_state)
    mask, fallback = candidate_mask_with_status(game_state, info, allowed(info, True))
    candidates = np.flatnonzero(mask)
    raw = averaged_values(self, info['key'])
    scores = raw - escape_costs(game_state, info, 0., 1.)
    best = candidates[np.isclose(scores[candidates], scores[candidates].max())]
    corrected = False
    # When the worst-case intersection is empty, preserving every static
    # candidate is necessary. Do not then let contested-tile cost alone turn
    # an otherwise preferred retreat into a unique WAIT. Keep distance cost,
    # the learned Q ranking, and the SAME safety mask. Ordinary choices and
    # any tie including a move are left exactly as before.
    if info['threatened'] and fallback and mask[:4].any() and np.array_equal(best, [4]):
        retreat_scores = raw - escape_costs(game_state, info, 0., 0.)
        retreat_best = candidates[np.isclose(retreat_scores[candidates], retreat_scores[candidates].max())]
        if np.all(retreat_best < 4):
            scores = retreat_scores
            corrected = True
    return mask, scores, corrected


def act(self, game_state):
    mask, scores, _ = scored_actions(self, game_state)
    candidates = np.flatnonzero(mask)
    best = candidates[np.isclose(scores[candidates], scores[candidates].max())]
    return ACTIONS[int(self.rng.choice(best))]
