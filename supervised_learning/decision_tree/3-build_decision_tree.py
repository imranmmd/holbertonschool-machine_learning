#!/usr/bin/env python3
"""
Decision Tree Components
Includes classes for nodes (both decision and leaf nodes) and the
decision tree itself.
"""
import numpy as np


class Node:
    """
    Represents a decision node in a decision tree.
    """

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """
        Initialize a decision node.
        """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """
        Return the maximum depth below this node.
        """
        max_depth = self.depth

        if self.left_child is not None:
            max_depth = max(
                max_depth, self.left_child.max_depth_below())

        if self.right_child is not None:
            max_depth = max(
                max_depth, self.right_child.max_depth_below())

        return max_depth

    def count_nodes_below(self, only_leaves=False):
        """
        Count the nodes below this node.

        Args:
            only_leaves (bool): If True, count only leaves.

        Returns:
            int: Number of nodes.
        """
        if only_leaves:
            count = 0
        else:
            count = 1

        if self.left_child is not None:
            count += self.left_child.count_nodes_below(
                only_leaves=only_leaves)

        if self.right_child is not None:
            count += self.right_child.count_nodes_below(
                only_leaves=only_leaves)

        return count

    def get_leaves_below(self):
        """
        Return a list of all leaves below this node.
        """
        leaves = []

        if self.left_child is not None:
            leaves.extend(
                self.left_child.get_leaves_below())

        if self.right_child is not None:
            leaves.extend(
                self.right_child.get_leaves_below())

        return leaves

    def __str__(self):
        """
        Return string representation of node.
        """
        node_type = "root" if self.is_root else "node"
        details = (f"{node_type} [feature={self.feature}, "
                   f"threshold={self.threshold}]\n")

        if self.left_child is not None:
            left_str = self.left_child.__str__().replace(
                "\n", "\n    |  ")
            details += f"    +---> {left_str}"

        if self.right_child is not None:
            right_str = self.right_child.__str__().replace(
                "\n", "\n       ")
            details += f"\n    +---> {right_str}"

        return details


class Leaf(Node):
    """
    Represents a leaf node in a decision tree.
    """

    def __init__(self, value, depth=None):
        """
        Initialize a leaf node.
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Return depth of the leaf.
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Return 1 for leaf.
        """
        return 1

    def get_leaves_below(self):
        """
        Return list containing this leaf.
        """
        return [self]

    def __str__(self):
        """
        Return string representation of leaf.
        """
        return f"-> leaf [value={self.value}] "


class Decision_Tree:
    """
    Decision Tree class.
    """

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """
        Initialize decision tree.
        """
        self.rng = np.random.default_rng(seed)

        if root is not None:
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
        """
        Return maximum depth of tree.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Count nodes in tree.
        """
        return self.root.count_nodes_below(
            only_leaves=only_leaves)

    def get_leaves(self):
        """
        Return all leaves of the tree.
        """
        return self.root.get_leaves_below()

    def __str__(self):
        """
        Return string representation of tree.
        """
        return self.root.__str__() + "\n"
