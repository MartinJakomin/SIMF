# SIMF

Simultaneous incremental matrix factorization: Python implementation.


## Dependencies:
Python 3
* numpy
* scipy
* (optional) matplotlib
* (optional) tables

## Usage:
```
from scipy import sparse
from simf import *

R1 = sparse.random(1000, 1000)
R2 = sparse.random(1000, 1000)

o1 = ObjectType('O1', 10)
o2 = ObjectType('O2', 5)
o3 = ObjectType('O3', 5)
r1 = Relation(o1, o2, R1, weight=1)
r2 = Relation(o1, o3, R2, weight=1)
data = [r1, r2]

model = SIMF()
model.fit(data)
error = model.get_train_error()
```

For more usages see [examples](/examples).


## Paper
See [Simultaneous incremental matrix factorization for streaming recommender systems](#).
