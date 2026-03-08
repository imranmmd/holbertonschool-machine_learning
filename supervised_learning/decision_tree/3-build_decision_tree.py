#!/usr/bin/env python3
"""Decision tree classes."""


class Node:
    """Represents an internal node of a decision tree."""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Initialize a node."""
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
        return ("-> {} [feature={}] [threshold={}]"
                .format(node_type, self.feature, self.threshold))

    def get_leaves_below(self):
        """Return the list of all leaves below the node."""
        leaves = []

        if self.left_child is not None:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child is not None:
            leaves.extend(self.right_child.get_leaves_below())

        return leaves


class Leaf(Node):
    """Represents a leaf of a decision tree."""

    def __init__(self, value, depth=None):
        """Initialize a leaf."""
        super().__init__(depth=depth)
        self.value = value
        self.is_leaf = True

    def __str__(self):
        """Return the string representation of the leaf."""
        return "-> leaf [value={}] ".format(self.value)

    def get_leaves_below(self):
        """Return the leaf itself in a list."""
        return [self]


class Decision_Tree:
    """Represents a decision tree."""

    def __init__(self, root=None):
        """Initialize a decision tree."""
        self.root = root

    def get_leaves(self):
        """Return the list of all leaves of the tree."""
        return self.root.get_leaves_below()
