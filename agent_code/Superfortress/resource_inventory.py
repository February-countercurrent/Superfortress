"""Public coin accounting for a known, fixed-size, non-respawning coin pool.

Design prototype only; not imported by any deployed policy. A missing observation
makes a positive hidden count an upper bound, while seeing every coin still proves
exhaustion. Do not pass hidden coin locations or final match statistics here.
"""


class CoinInventory:
    def __init__(self, total_coins=None):
        if total_coins is not None and (not isinstance(total_coins, int) or total_coins < 0):
            raise ValueError('Coin total must be a known nonnegative integer or None')
        self.total = total_coins
        self.round = None
        self.step = 0
        self.continuous = True
        self.seen = set()

    def observe(self, state):
        if state['round'] != self.round:
            self.round = state['round']
            self.step = 0
            self.continuous = True
            self.seen.clear()
        step = int(state['step'])
        if step <= self.step:
            raise ValueError('Observe each decision once, in increasing order')
        if step != self.step + 1:
            self.continuous = False
        visible = set(map(tuple, state['coins']))
        self.seen.update(visible)
        if self.total is not None and len(self.seen) > self.total:
            raise ValueError('Observed more coins than the configured fixed pool')
        self.step = step
        upper = None if self.total is None else self.total - len(self.seen)
        return dict(total_coins=self.total, ever_seen=len(self.seen), visible=len(visible),
                    observation_continuous=self.continuous,
                    hidden_upper_bound=upper,
                    hidden_exact=upper if self.continuous else (0 if upper == 0 else None),
                    no_hidden_coins=upper == 0,
                    no_resource_coins=upper == 0 and not visible)
