#!/usr/bin/env python3
"""Module that defines a random forest."""


Decision_Tree = __import__('8-build_decision_tree').Decision_Tree
import numpy as np


class Random_Forest:
    """Class that represents a random forest."""

    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """Initialize a Random_Forest."""
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed

    def predict(self, explanatory):
        """Predict classes for all rows in explanatory."""
        predictions = np.array(
            [predict(explanatory) for predict in self.numpy_preds]
        )
        values = np.unique(predictions)
        counts = np.array(
            [np.sum(predictions == value, axis=0) for value in values]
        )
        return values[np.argmax(counts, axis=0)]

    def fit(self, explanatory, target, n_trees=100, verbose=0):
        """Train the random forest."""
        self.target = target
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        accuracies = []

        for i in range(n_trees):
            tree = Decision_Tree(
                max_depth=self.max_depth,
                min_pop=self.min_pop,
                seed=self.seed + i
            )
            tree.fit(explanatory, target)
            self.numpy_preds.append(tree.predict)
            depths.append(tree.depth())
            nodes.append(tree.count_nodes())
            leaves.append(tree.count_nodes(only_leaves=True))
            accuracies.append(tree.accuracy(tree.explanatory, tree.target))

        if verbose == 1:
            print(
                """  Training finished.
    - Mean depth                     : {}
    - Mean number of nodes           : {}
    - Mean number of leaves          : {}
    - Mean accuracy on training data : {}
    - Accuracy of the forest on td   : {}""".format(
                    np.array(depths).mean(),
                    np.array(nodes).mean(),
                    np.array(leaves).mean(),
                    np.array(accuracies).mean(),
                    self.accuracy(self.explanatory, self.target)
                )
            )

    def accuracy(self, test_explanatory, test_target):
        """Compute prediction accuracy."""
        return np.sum(
            np.equal(self.predict(test_explanatory), test_target)
        ) / test_target.size
