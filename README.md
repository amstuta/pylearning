## Pylearning: python machine learning library

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/amstuta/pylearning/blob/master/LICENSE.md)
[![PyPI](https://img.shields.io/pypi/pyversions/pylearning.svg)]()

Pylearning is a high-level machine learning package designed to easily prototype
and implement data analysis programs.

The library includes the following algorithms:
- Regression:
    - Decision tree regressor
    - Random forest regressor
    - Nearest neighbours regressor
- Classification:
    - Decision tree classifier
    - Random forest classifier
    - Nearest neighbours classifier

The two random forests algorithms use multithreading to train the trees in a
parallelized fashion.
This package is compatible with Python3+.

### Basic usage

All the algorithms available use the same simple interface described in the
examples below.

```python
# Basic classification example using a decision tree

from pylearning.trees import DecisionTreeClassifier

# Load your training dataset
features, targets = ...

tree = DecisionTreeClassifier(max_depth=10)
tree.fit(features, targets)

# Load a testing sample
test_feature, test_target = ...

predicted_class = tree.predict(test_feature, test_target)
```

```python
# Basic regression example using a random forest

from pylearning.ensembles import RandomForestRegressor

# Load the training dataset
features, targets = ...

rf = RandomForestRegressor(nb_trees=10, nb_samples=100, max_depth=20)
rf.fit(features, targets)

# Load a testing sample
test_feature, test_target = ...

value_predicted = rf.predict(test_feature, test_target)
```

A complete documentation of the API is available [here](https://pythonhosted.org/pylearning/).

### Installation

Pylearning requires to have numpy installed. It can be installed simply using Pypy:
```sh
# for the stable version
pip3 install pylearning

# for the latest version
pip3 install git+https://github.com/amstuta/pylearning.git
```

### Further improvements

The core functionalities of trees, random forest and nearest neighbours are
implemented in this project, however there are many improvements that could be
added:
- gini criterion for splitting nodes
- pruning
- ability to split a node into an arbitrary number of child nodes
- optimizations to reduce time and memory consumption
- better compatibility with pandas DataFrame
- ...

If you wish, you're welcome to participate in the project or to make suggestions !
To do so, you can simply open an issue or fork the project and then create a pull
request.
