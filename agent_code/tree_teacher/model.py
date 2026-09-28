"""Shared or action-specific Extra Trees with identical D4 value semantics."""
from functools import lru_cache
from pathlib import Path
import pickle
import numpy as np
from .features import ACTIONS
from .symmetry_maps import PERMUTATIONS


def transform_keys(keys, permutation):
    keys = np.asarray(keys,dtype=np.float32)
    if keys.ndim != 2 or keys.shape[1] not in (16,28):
        raise ValueError('Expected a batch of 16- or 28-feature states')
    result = keys.copy()
    groups = (0,6) if keys.shape[1] == 16 else (0,6,18,22)
    for start in groups:
        for old in range(4): result[:,start+permutation[old]] = keys[:,start+old]
    return result


def state_actions(keys,actions):
    keys = np.asarray(keys,dtype=np.float32)
    actions = np.asarray(actions,dtype=int)
    if actions.shape != (len(keys),) or ((actions<0)|(actions>=6)).any():
        raise ValueError('Invalid action batch')
    return np.column_stack((keys,np.eye(6,dtype=np.float32)[actions]))


def predict_chunks(forest,inputs):
    return np.concatenate([forest.predict(inputs[i:i+65536]) for i in range(0,len(inputs),65536)])


class D4Queries:
    def __init__(self,keys):
        keys=np.asarray(keys,dtype=np.float32)
        self.count=len(keys)
        transformed=np.concatenate([transform_keys(keys,p) for p in PERMUTATIONS])
        self.keys,inverse=np.unique(transformed,axis=0,return_inverse=True)
        self.inverse=inverse.reshape(8,self.count)
        self.shared_inputs=None

    def predict(self,estimators):
        if estimators is None or self.count==0:
            return np.zeros((self.count,6),dtype=float)
        if isinstance(estimators,list):
            if len(estimators)!=6: raise ValueError('One regressor per action is required')
            raw=np.column_stack([predict_chunks(f,self.keys) for f in estimators])
        else:
            if self.shared_inputs is None:
                self.shared_inputs=state_actions(np.repeat(self.keys,6,axis=0),np.tile(np.arange(6),len(self.keys)))
            raw=predict_chunks(estimators,self.shared_inputs).reshape(-1,6)
        values=np.mean([raw[self.inverse[g]][:,list(p)] for g,p in enumerate(PERMUTATIONS)],axis=0)
        if not np.isfinite(values).all(): raise FloatingPointError('Non-finite Q values')
        return values


class ForestQ:
    def __init__(self,estimators,features):
        self.estimators,self.features=estimators,features
        self._cached=lru_cache(maxsize=8192)(self._one)

    def _one(self,key):
        if len(key)!=self.features: raise ValueError('Feature/checkpoint mismatch')
        return tuple(D4Queries([key]).predict(self.estimators)[0])

    def values(self,key):
        return np.asarray(self._cached(tuple(key)),dtype=float)


def save_model(path,estimators,features,kind,**metadata):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    value=dict(version=2,features=features,kind=kind,actions=list(ACTIONS),d4_mean=True,
               estimators=estimators,**metadata)
    temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as handle: pickle.dump(value,handle,protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def load_model(path):
    with Path(path).open('rb') as handle: data=pickle.load(handle)
    if (data['version']!=2 or data['features'] not in (16,28) or
        data['kind'] not in ('shared','heads') or data['actions']!=list(ACTIONS) or not data['d4_mean']):
        raise ValueError('Incompatible tactical tree checkpoint')
    estimators=data['estimators']
    if data['kind']=='heads':
        if not isinstance(estimators,list) or len(estimators)!=6: raise ValueError('Expected six regressors')
        forests=estimators; expected=data['features']
    else:
        if isinstance(estimators,list): raise ValueError('Expected one shared regressor')
        forests=[estimators]; expected=data['features']+6
    for forest in forests:
        if forest.n_features_in_!=expected: raise ValueError('Wrong estimator input size')
        forest.n_jobs=1
    return ForestQ(estimators,data['features']),{k:v for k,v in data.items() if k!='estimators'}
