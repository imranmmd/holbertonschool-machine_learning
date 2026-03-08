#!/usr/bin/env python3
"""Module that defines and trains a simple decision tree."""


import numpy as np


class Node:
    """Class that represents an internal node of a decision tree."""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Initialize a Node."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_root = is_root
        self.depth = depth
        self.is_leaf = False

    def __str__(self):
        """Return the string representation of the node."""
        node_type = "root" if self.is_root else "node"
        return "{} [feature={}, threshold={}]".format(
            node_type, self.feature, self.threshold
        )

    def get_leaves_below(self):
        """Return the list of all leaves below this node."""
        leaves = []

        if self.left_child is not None:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child is not None:
            leaves.extend(self.right_child.get_leaves_below())

        return leaves

    def update_bounds_below(self):
        """Update bounds dictionaries for all nodes below this node."""
        if self.is_root:
            n_features = self.left_child.depth if False else None
            self.lower = getattr(self, "lower", {0: -1 * np.inf})
            self.upper = getattr(self, "upper", {0: np.inf})

        self.left_child.lower = self.lower.copy()
        self.left_child.upper = self.upper.copy()
        self.left_child.lower[self.feature] = self.threshold

        self.right_child.lower = self.lower.copy()
        self.right_child.upper = self.upper.copy()
        self.right_child.upper[self.feature] = self.threshold

        self.left_child.update_bounds_below()
        self.right_child.update_bounds_below()

    def update_indicator(self):
        """Compute and store the indicator function of the node."""

        def is_large_enough(x):
            """Check whether individuals satisfy all lower bounds."""
            return np.all(
                np.array(
                    [
                        np.greater(x[:, key], self.lower[key])
                        for key in self.lower.keys()
                    ]
                ),
                axis=0
            )

        def is_small_enough(x):
            """Check whether individuals satisfy all upper bounds."""
            return np.all(
                np.array(
                    [
                        np.less_equal(x[:, key], self.upper[key])
                        for key in self.upper.keys()
                    ]
                ),
                axis=0
            )

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]),
            axis=0
        )

    def pred(self, x):
        """Predict the class for one individual."""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        return self.right_child.pred(x)


class Leaf(Node):
    """Class that represents a leaf of a decision tree."""

    def __init__(self, value, depth=None):
        """Initialize a Leaf."""
        super().__init__(depth=depth)
        self.value = value
        self.is_leaf = True

    def __str__(self):
        """Return the string representation of the leaf."""
        return "-> leaf [value={}]".format(self.value)

    def get_leaves_below(self):
        """Return a list containing this leaf."""
        return [self]

    def update_bounds_below(self):
        """Do nothing for a leaf."""
        pass

    def pred(self, x):
        """Predict the class for one individual."""
        return self.value


class Decision_Tree:
    """Class that represents a decision tree."""

    def __init__(self, root=None, split_criterion="random", max_depth=10,
                 min_pop=1, seed=0):
        """Initialize a Decision_Tree."""
        if root is None:
            root = Node(is_root=True, depth=0)
        self.root = root
        self.split_criterion = split_criterion
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.rng = np.random.default_rng(seed)

    def __str__(self):
        """Return the string representation of the tree."""
        lines = []
        self._stringify(self.root, lines, "")
        return "\n".join(lines)

    def _stringify(self, node, lines, prefix):
        """Build recursively the string representation of the tree."""
        if node.is_root:
            lines.append(str(node))
        else:
            lines.append(prefix + str(node))

        if node.is_leaf:
            return

        self._stringify(node.left_child, lines, prefix + "    +---> ")
        self._stringify(node.right_child, lines, prefix + "    +---> ")

    def get_leaves(self):
        """Return the list of all leaves of the tree."""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Update bounds for the whole tree."""
        n_features = self.explanatory.shape[1]
        self.root.lower = {i: -1 * np.inf for i in range(n_features)}
        self.root.upper = {i: np.inf for i in range(n_features)}
        self.root.update_bounds_below()

    def update_predict(self):
        """Compute and store the prediction function."""
        self.update_bounds()
        leaves = self.get_leaves()

        for leaf in leaves:
            leaf.update_indicator()

        self.predict = lambda a: np.sum(
            np.array(
                [
                    leaf.indicator(a) * leaf.value
                    for leaf in leaves
                ]
            ),
            axis=0
        )

    def pred(self, x):
        """Predict the class for one individual."""
        return self.root.pred(x)

    def np_extrema(self, arr):
        """Return the minimum and maximum of a NumPy array."""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Return a random feature and threshold for a node."""
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            values = self.explanatory[:, feature][node.sub_population]
            feature_min, feature_max = self.np_extrema(values)
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def fit(self, explanatory, target, verbose=0):
        """Train the decision tree."""
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.random_split_criterion

        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype="bool")

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(
                """  Training finished.
    - Depth                     : {}
    - Number of nodes           : {}
    - Number of leaves          : {}
    - Accuracy on training data : {}""".format(
                    self.depth(),
                    self.count_nodes(),
                    self.count_nodes(only_leaves=True),
                    self.accuracy(self.explanatory, self.target)
                )
            )

    def fit_node(self, node):
        """Recursively fit a node and its children."""
        node.feature, node.threshold = self.split_criterion(node)

        feature_values = self.explanatory[:, node.feature]
        current_pop = node.sub_population

        left_population = np.logical_and(
            current_pop,
            np.greater(feature_values, node.threshold)
        )
        right_population = np.logical_and(
            current_pop,
            np.less_equal(feature_values, node.threshold)
        )

        is_left_leaf = self._is_leaf(left_population, node.depth + 1)
        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        is_right_leaf = self._is_leaf(right_population, node.depth + 1)
        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def _is_leaf(self, sub_population, depth):
        """Return whether a child should be a leaf."""
        pop_size = np.sum(sub_population)
        sub_target = self.target[sub_population]

        if pop_size <= self.min_pop:
            return True
        if depth >= self.max_depth:
            return True
        if np.unique(sub_target).size == 1:
            return True
        return False

    def get_leaf_child(self, node, sub_population):
        """Return a fitted leaf child."""
        classes, counts = np.unique(
            self.target[sub_population],
            return_counts=True
        )
        value = classes[np.argmax(counts)]
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Return a non-leaf child node."""
        child = Node()
        child.depth = node.depth + 1
        child.sub_population = sub_population
        return child

    def accuracy(self, test_explanatory, test_target):
        """Compute prediction accuracy."""
        return np.sum(
            np.equal(self.predict(test_explanatory), test_target)
        ) / test_target.size

    def count_nodes(self, only_leaves=False):
        """Count nodes in the tree."""
        return self._count_nodes_below(self.root, only_leaves)

    def _count_nodes_below(self, node, only_leaves=False):
        """Count nodes below a given node."""
        if node is None:
            return 0
        if node.is_leaf:
            return 1
        if only_leaves:
            return (
                self._count_nodes_below(node.left_child, only_leaves) +
                self._count_nodes_below(node.right_child, only_leaves)
            )
        return 1 + (
            self._count_nodes_below(node.left_child, only_leaves) +
            self._count_nodes_below(node.right_child, only_leaves)
        )

    def depth(self):
        """Return the depth of the tree."""
        return self._depth_below(self.root)

    def _depth_below(self, node):
        """Return the maximum depth below a node."""
        if node.is_leaf:
            return node.depth
        return max(
            self._depth_below(node.left_child),
            self._depth_below(node.right_child)
        )
