#!/usr/bin/env python3
"""Module that defines a simple decision tree."""


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
            self.upper = {0: np.inf}
            self.lower = {0: -1 * np.inf}

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
        """Update bounds below the leaf."""
        pass

    def pred(self, x):
        """Predict the class for one individual."""
        return self.value


class Decision_Tree:
    """Class that represents a decision tree."""

    def __init__(self, root=None):
        """Initialize a Decision_Tree."""
        self.root = root

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

        self._stringify(
            node.left_child,
            lines,
            prefix + "    +---> "
        )
        self._stringify(
            node.right_child,
            lines,
            prefix + "    +---> "
        )

    def get_leaves(self):
        """Return the list of all leaves of the tree."""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Update bounds for the whole tree."""
        self.root.update_bounds_below()

    def update_predict(self):
        """Compute and store the prediction function."""
        self.update_bounds()
        leaves = self.get_leaves()

        for leaf in leaves:
            leaf.update_indicator()

        self.predict = lambda A: np.sum(
            np.array(
                [
                    leaf.indicator(A) * leaf.value
                    for leaf in leaves
                ]
            ),
            axis=0
        )

    def pred(self, x):
        """Predict the class for one individual."""
        return self.root.pred(x)
