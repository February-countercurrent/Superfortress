"""One-opponent, one-move escape robustness filter; preserves learned features."""
import numpy as np
from .features import DELTAS, forecast


def candidate_mask(state, info, base):
    return candidate_mask_with_status(state, info, base)[0]


def candidate_mask_with_status(state, info, base):
    """Return the unchanged deployed mask and whether intersection failed.

    A legal-only fallback because no static escape exists is NOT flagged as
    a dynamic intersection failure. The caller must not treat it as safe.
    """
    if not info['safe'].any() or not state['others']:
        return base.copy(), False
    # Only affect bomb placement or responses to existing blast threats.
    if not info['threatened'] and not base[5]:
        return base.copy(), False
    position = tuple(state['self'][3])
    occupied = {tuple(o[3]) for o in state['others']} | {tuple(p) for p, _ in state['bombs']} | {position}
    destinations = [(position[0] + dx, position[1] + dy) for dx, dy in DELTAS] + [position, position]
    robust = base.copy()
    for i, other in enumerate(state['others']):
        ox, oy = other[3]
        if abs(ox - position[0]) + abs(oy - position[1]) > 6:
            continue
        for dx, dy in DELTAS:
            p = ox + dx, oy + dy
            if not (0 <= p[0] < state['field'].shape[0] and 0 <= p[1] < state['field'].shape[1]):
                continue
            if state['field'][p] != 0 or p in occupied:
                continue
            others = list(state['others'])
            others[i] = (*other[:3], p)
            predicted = {**state, 'others': others}
            if info['threatened']:
                viable = forecast(predicted)
                for action in range(5):
                    # If the opponent arrives first, our move fails and we stay.
                    destination = position if action < 4 and destinations[action] == p else destinations[action]
                    robust[action] &= viable[destination]
            if robust[5]:
                robust[5] &= forecast(predicted, new_bomb=True)[position]
    # Avoid replacing an existing safe escape with an unsafe move if the
    # conservative scenarios eliminate everything. Q policy handles fallback.
    return (robust, False) if robust.any() else (base.copy(), True)
