#!/usr/bin/env python3
"""Module that defines an isolation random tree."""


import numpy as np

Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf


class Isolation_Random_Tree:
    """Class that represents an isolation random tree."""

    def __init__(self, max_depth=10, seed=0, root=None):
        """Initialize an Isolation_Random_Tree."""
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.max_depth = max_depth
        self.predict = None
        self.min_pop = 1

    def __str__(self):
        """Return the string representation of the tree."""
        lines = []
        self._stringify(self.root, lines, "", True)
        return "\n".join(lines)

    def _stringify(self, node, lines, prefix, is_last):
        """Build recursively the string representation of the tree."""
        if node.is_root:
            lines.append(str(node))
        else:
            lines.append(prefix + "+---> " + str(node))

        if node.is_leaf:
            return

        child_prefix = prefix
        if not node.is_root:
            if is_last:
                child_prefix += "       "
            else:
                child_prefix += "|      "

        self._stringify(node.left_child, lines, child_prefix, False)
        self._stringify(node.right_child, lines, child_prefix, True)

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

    def update_bounds(self):
        """Update bounds for the whole tree."""
        n_features = self.explanatory.shape[1]
        self.root.lower = {i: -1 * np.inf for i in range(n_features)}
        self.root.upper = {i: np.inf for i in range(n_features)}
        self.root.update_bounds_below()

    def get_leaves(self):
        """Return the list of all leaves of the tree."""
        return self.root.get_leaves_below()

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

    def get_leaf_child(self, node, sub_population):
        """Return a leaf child whose value is its depth."""
        leaf_child = Leaf(node.depth + 1)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Return a non-leaf child node."""
        child = Node()
        child.depth = node.depth + 1
        child.sub_population = sub_population
        return child

    def fit_node(self, node):
        """Recursively fit a node and its children."""
        node.feature, node.threshold = self.random_split_criterion(node)

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

        is_left_leaf = (
            np.sum(left_population) <= self.min_pop or
            node.depth + 1 >= self.max_depth
        )

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        is_right_leaf = (
            np.sum(right_population) <= self.min_pop or
            node.depth + 1 >= self.max_depth
        )

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):
        """Train the isolation random tree."""
        self.split_criterion = self.random_split_criterion
        self.explanatory = explanatory
        self.root.sub_population = np.ones(
            explanatory.shape[0],
            dtype='bool'
        )

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(
                """  Training finished.
    - Depth                     : {}
    - Number of nodes           : {}
    - Number of leaves          : {}""".format(
                    self.depth(),
                    self.count_nodes(),
                    self.count_nodes(only_leaves=True)
                )
            )
