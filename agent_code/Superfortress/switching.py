"""Two independently switchable guards around the unchanged goal controller."""
import numpy as np

from .legacy_switching import SwitchController as LegacyController
from .features import ACTIONS, blast
from .resource_proposal_guard import resource_evidence, resolve_proposal


class SwitchController(LegacyController):
    def __init__(self, inventory_guard=False, bomb_guard=False):
        super().__init__()
        self.inventory_guard = inventory_guard
        self.bomb_guard = bomb_guard

    def select_guarded(self, state, info, mask, scores, ledger, history, rng,
                       inventory, baseline):
        options, details = super().select(state, info, mask, scores, ledger, history, rng)
        details.update(guard=None, accepted=False)
        commitment = details.get('commitment')
        if commitment is None:
            return options, details
        b = sum(state['field'][p] == 1 for p in blast(state['field'], commitment['target']))
        evidence = resource_evidence(commitment['kind'], inventory, int(b),
                                     int(np.count_nonzero(state['field'] == 1)))
        details['evidence'] = evidence
        reason = None
        if self.inventory_guard and not evidence['eligible']:
            reason = evidence['reason']
        elif self.bomb_guard and options == [5]:
            # The original controller only proposes BOMB at a crate goal.
            _, verdict = resolve_proposal(ACTIONS[baseline], 'BOMB',
                [ACTIONS[i] for i in np.flatnonzero(mask)], dict(eligible=True))
            if verdict != 'proposal_accepted':
                reason = verdict
        if reason:
            details.update(guard=reason, reason='guard_rejected', ended=commitment.copy(), allowed=[])
            # An attempted commitment consumes the same two-cycle evidence as
            # legacy activation, even if immediately rejected. Do not retry it
            # on every subsequent step without two fresh completed cycles.
            self.active = None
            return [], details
        details['accepted'] = True
        return options, details
