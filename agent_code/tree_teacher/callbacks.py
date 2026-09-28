"""Frozen tactical FQI plus a bounded WAIT-to-retreat experiment."""
import json
import os
from pathlib import Path
import numpy as np
from .features import ACTIONS,allowed,describe
from .dynamic import candidate_mask
from .tactical import tactical_key
from .model import load_model
from .escape_wait import retreat_options


def setup(self):
    if self.train: raise ValueError('Fit tactical trees offline using the experiment entry point')
    path=Path(os.environ.get('FQI_MODEL',str(Path(__file__).parent/'model.pkl')))
    self.predictor,self.metadata=load_model(path)
    self.seed=int(os.environ.get('FQI_SEED','42'))
    self.policy_round=None
    self.escape_enabled=os.environ.get('FQI_ESCAPE_GUARD','1')=='1'
    trace=os.environ.get('FQI_POLICY_TRACE')
    self.policy_trace=Path(trace).open('x',encoding='utf-8') if trace else None


def act(self,game_state):
    if self.policy_round!=game_state['round']:
        self.policy_round=game_state['round']
        self.rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.policy_round,101]))
        self.guard_rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.policy_round,106]))
    info=describe(game_state)
    mask=candidate_mask(game_state,info,allowed(info,True))
    if not mask.any(): raise ValueError('Empty dynamic candidate mask')
    key=info['key'] if self.predictor.features==16 else tactical_key(game_state,info)
    values=self.predictor.values(tuple(key))
    candidates=np.flatnonzero(mask)
    best=candidates[np.isclose(values[candidates],values[candidates].max())]
    baseline=int(self.rng.choice(best))
    options,distances=retreat_options(game_state,info,mask,values) if self.escape_enabled else ([],None)
    selected=int(self.guard_rng.choice(options)) if options else baseline
    assert mask[selected]
    if selected!=baseline:
        assert baseline==4 and selected<4 and info['threatened'] and info['safe'][selected]
    if self.policy_trace:
        self.policy_trace.write(json.dumps(dict(round=game_state['round'],step=game_state['step'],
            baseline=baseline,selected=selected,changed=selected!=baseline,options=options,
            distances=distances,threatened=bool(info['threatened']),safe=info['safe'].tolist(),
            mask=mask.tolist(),q=values.tolist()))+'\n')
    return ACTIONS[selected]
