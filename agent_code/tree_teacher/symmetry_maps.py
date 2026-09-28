"""D4 transformations of the legacy 16-feature observation."""
PERMUTATIONS = tuple(tuple((r + sign*i) % 4 for i in range(4)) + (4,5)
                     for sign in (1,-1) for r in range(4))


def transform_key(key, permutation):
    if len(key) != 16:
        raise ValueError('Expected original 16-feature observation')
    result=list(key)
    for old in range(4):
        result[permutation[old]]=key[old]
        result[6+permutation[old]]=key[6+old]
    return tuple(result)
