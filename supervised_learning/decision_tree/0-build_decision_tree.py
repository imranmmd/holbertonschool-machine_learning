#!/usr/bin/env python3
"""Decision Tree module."""

import numpy as np


class Node:
    """Represents an internal node of a decision tree."""

    def __init__(self, feature=None, threshold=None,
                 left_child=None, right_child=None,
                 is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Return maximum depth below this node."""
        if self.left_child is None and self.right_child is None:
            return self.depth

        left_depth = self.depth
        right_depth = self.depth

        if self.left_child is not None:
            left_depth = self.left_child.max_depth_below()

        if self.right_child is not None:
            right_depth = self.right_child.max_depth_below()

        return max(left_depth, right_depth)


class Leaf(Node):
    """Represents a leaf in the decision tree."""

    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Return depth of leaf."""
        return self.depth


class Decision_Tree:
    """Decision Tree class."""

    def __init__(self, max_depth=10, min_pop=1,
                 seed=0, split_criterion="random",
                 root=None):
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)

        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """Return depth of the tree."""
        return self.root.max_depth_below()
