# SIMF - Simultaneous Incremental Matrix Factorization


## Dependencies:

The `simf` module runs in Python3 and requires the following modules:

* numpy
* scipy
* matplotlib (optional)
* tables (optional)


## Quick-start
Get the latest version of `simf` from `github`:

    git clone https://github.com/MartinJakomin/SIMF.git


Install it in development mode:

    cd SIMF
    pip install -e .


Test the installation:

    python -c "import simf"


## Jupter notebook on mybinder.org
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/MartinJakomin/SIMF/master?filepath=examples)

Try `simf` by running it on Binder.


## Use
Start python and then run the following lines:

    import simf
    from scipy import sparse

    R1 = sparse.random(1000, 1000)
    R2 = sparse.random(1000, 1000)

    o1 = simf.ObjectType('O1', 10)
    o2 = simf.ObjectType('O2', 5)
    o3 = simf.ObjectType('O3', 5)
    r1 = simf.Relation(o1, o2, R1, weight=1)
    r2 = simf.Relation(o1, o3, R2, weight=1)
    data = [r1, r2]

    model = simf.SIMF()
    model.fit(data)
    error = model.get_train_error()


For more use cases, see the [examples](/examples) folder.

Some of the graphs from the paper can be recreated by running the
[`yelp.ipynb`](https://mybinder.org/v2/gh/MartinJakomin/SIMF/master?filepath=examples%2Fyelp.ipynb) notebook.


## How to cite

    @article{jakomin2019,
        title     = {Simultaneous incremental matrix factorization for streaming recommender systems},
        author    = {Jakomin, Martin and Bosnić, Zoran and Curk, Tomaž},
        journal   = {},
        volume    = {},
        pages     = {},
        year      = {2018},
        note      = {Manuscript submitted for publication.},
    }

