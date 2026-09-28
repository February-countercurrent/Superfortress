"""Reviewable resource-branch guards; not connected to the deployed agent.

Exhaustion only vetoes a resource proposal. It does not veto the underlying
Q policy's attack or crate decisions. No new reward or probability is invented.
"""


def resource_evidence(kind, inventory, blast_crates=0, remaining_crates=0):
    if kind == 'coin':
        return dict(eligible=inventory['visible'] > 0, reason='visible_coin' if inventory['visible'] else 'no_visible_coin',
                    expected_revealed_coins=None)
    if kind != 'crate':
        raise ValueError('Expected coin or crate proposal')
    hidden = inventory['hidden_exact']
    if inventory['no_hidden_coins']:
        return dict(eligible=False, reason='hidden_pool_exhausted', expected_revealed_coins=0.)
    if hidden is None:
        return dict(eligible=False, reason='hidden_pool_unknown', expected_revealed_coins=None)
    if remaining_crates <= 0 or blast_crates <= 0:
        return dict(eligible=False, reason='no_crate_yield', expected_revealed_coins=0.)
    if not 0 <= hidden <= remaining_crates or not 0 <= blast_crates <= remaining_crates:
        raise ValueError('Inconsistent inventory')
    return dict(eligible=True, reason='possible_hidden_resource',
                expected_revealed_coins=hidden * blast_crates / remaining_crates)


def resolve_proposal(baseline_action, proposed_action, allowed_actions, evidence):
    """A goal branch may steer movement, but cannot introduce a new BOMB."""
    if not evidence['eligible']:
        return baseline_action, evidence['reason']
    if proposed_action not in allowed_actions:
        return baseline_action, 'outside_existing_mask'
    if proposed_action == 'BOMB' and baseline_action != 'BOMB':
        return baseline_action, 'resource_branch_cannot_force_bomb'
    return proposed_action, 'proposal_accepted'
